#TODO:
# batch by vad
# figure out stft shift
# upstream ctc changes
# gpu levenshtein, needleman/hirschberg

import os
import time
import json
import argparse
import importlib
import torch
import torch.nn.functional as F
import datasets
import models
import metrics
import decoders
import ctc
import vad
import transcripts
import audio
import vis

def setup(args):
	torch.set_grad_enabled(False)
	checkpoint = torch.load(args.checkpoint, map_location = 'cpu')
	args.sample_rate, args.window_size, args.window_stride, args.window, args.num_input_features = map(checkpoint['args'].get, ['sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features'])
	frontend = models.LogFilterBankFrontend(args.num_input_features, args.sample_rate, args.window_size, args.window_stride, args.window, eps = 1e-6)
	labels = datasets.Labels(datasets.Language(checkpoint['args']['lang']), name = 'char')
	model = getattr(models, args.model or checkpoint['args']['model'])(args.num_input_features, [len(labels)], frontend = frontend, dict = lambda logits, log_probs, olen, **kwargs: (logits[0], olen[0]))
	model.load_state_dict(checkpoint['model_state_dict'], strict = False)
	model = model.to(args.device)
	model.eval()
	model.fuse_conv_bn_eval()
	if args.device != 'cpu':
		model, *_ = models.data_parallel_and_autocast(model, opt_level = args.fp16)
	decoder = decoders.GreedyDecoder() if args.decoder == 'GreedyDecoder' else decoders.BeamSearchDecoder(labels, lm_path = args.lm, beam_width = args.beam_width, beam_alpha = args.beam_alpha, beam_beta = args.beam_beta, num_workers = args.num_workers, topk = args.decoder_topk)
	return labels, frontend, model, decoder

def main(args):
	os.makedirs(args.output_path, exist_ok = True)
	data_paths = [p for f in args.input_path for p in ([os.path.join(f, g) for g in os.listdir(f)] if os.path.isdir(f) else [f]) if os.path.isfile(p) and any(map(p.endswith, args.ext))] + [p for p in args.input_path if any(map(p.endswith, ['.json', '.json.gz']))]
	exclude = set([os.path.splitext(basename)[0] for basename in os.listdir(args.output_path) if basename.endswith('.json')] if args.skip_processed else [])
	data_paths = [path for path in data_paths if os.path.basename(path) not in exclude]

	labels, frontend, model, decoder = setup(args)
	val_dataset = datasets.AudioTextDataset(data_paths, [labels], args.sample_rate, frontend = None, segmented = True, mono = args.mono, time_padding_multiple = args.batch_time_padding_multiple, audio_backend = args.audio_backend, speakers = args.speakers, exclude=exclude, shuffle=args.shuffle)
	num_examples = len(val_dataset.examples)
	print("Examples count: ", num_examples)
	val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size = None, collate_fn = val_dataset.collate_fn, num_workers = args.num_workers)

	for i, (meta, x, xlen, y, ylen) in enumerate(val_data_loader):
		print(f'Processing: {i}/{num_examples}')

		audio_path, speakers = map(meta[0].get, ['audio_path', 'speakers'])

		duration = max(transcripts.compute_duration(t, hours = True) for t in meta)
		if x.numel() == 0 or (args.skip_file_longer_than_hours and duration > args.skip_file_longer_than_hours):
			print(f'Skipping [{audio_path}]. Size: {x.numel()}, duration: {duration} hours (>{args.skip_file_longer_than_hours})')
			continue

		transcript_path = os.path.join(args.output_path, os.path.basename(audio_path) + '.json')

		if max(ylen) > args.max_ref_len:
			print(f'Too large refs [{ylen}] [{audio_path}]. Skipping.')
			continue

		tic = time.time()
		y, ylen = y.to(args.device), ylen.to(args.device)
		log_probs, olen = model(x.squeeze(1).to(args.device), xlen.to(args.device))

		#speech = vad.detect_speech(x.squeeze(1), args.sample_rate, args.window_size, aggressiveness = args.vad, window_size_dilate = args.window_size_dilate)
		#speech = vad.upsample(speech, log_probs)
		#log_probs.masked_fill_(models.silence_space_mask(log_probs, speech, space_idx = labels.space_idx, blank_idx = labels.blank_idx), float('-inf'))
		
		decoded = decoder.decode(log_probs, olen)

		print('Input:', os.path.basename(audio_path))
		print('Input time steps:', log_probs.shape[-1], '| target time steps:', y.shape[-1])
		print('Time: audio {audio:.02f} sec | processing {processing:.02f} sec'.format(audio = sum(transcripts.compute_duration(t) for t in meta), processing =time.time() - tic))

		ts = (x.shape[-1] / args.sample_rate) * torch.linspace(0, 1, steps = log_probs.shape[-1]).unsqueeze(0) + torch.FloatTensor([t['begin'] for t in meta]).unsqueeze(1)
		channel = [t['channel'] for t in meta]
		speaker = [t['speaker'] for t in meta]
		ref_segments = [[dict(channel = channel[i], begin = meta[i]['begin'], end = meta[i]['end'], ref = labels.decode(y[i, 0, :ylen[i]].tolist()))] for i in range(len(decoded))]
		hyp_segments = [labels.decode(decoded[i], ts[i], channel = channel[i], replace_blank = True, replace_blank_series = args.replace_blank_series, replace_repeat = True, replace_space = False, speaker = speaker[i] if isinstance(speaker[i], str) else None) for i in range(len(decoded))]
		
		ref, hyp = '\n'.join(transcripts.join(ref = r) for r in ref_segments).strip(), '\n'.join(transcripts.join(hyp = h) for h in hyp_segments).strip()
		if args.verbose:
			print('HYP:', hyp)
		print('CER: {cer:.02%}'.format(cer = metrics.cer(hyp, ref)))

		tic_alignment = time.time()
		if args.align and y.numel() > 0:
			#if ref_full:# and not ref:
			#	#assert len(set(t['channel'] for t in meta)) == 1 or all(t['type'] != 'channel' for t in meta)
			#	#TODO: add space at the end
			#	channel = torch.ByteTensor(channel).repeat_interleave(log_probs.shape[-1]).reshape(1, -1)
			#	ts = ts.reshape(1, -1)
			#	log_probs = log_probs.transpose(0, 1).unsqueeze(0).flatten(start_dim = -2)
			#	olen = torch.tensor([log_probs.shape[-1]], device = log_probs.device, dtype = torch.long)
			#	y = y_full[None, None, :].to(y.device)
			#	ylen = torch.tensor([[y.shape[-1]]], device = log_probs.device, dtype = torch.long)
			#	segments = [([], sum([h for r, h in segments], []))]
			
			alignment = ctc.alignment(log_probs.permute(2, 0, 1), y.squeeze(1), olen, ylen.squeeze(1), blank = labels.blank_idx)
			ref_segments = [labels.decode(y[i, 0, :ylen[i]].tolist(), ts[i], alignment[i], channel = channel[i], speaker = speaker[i], key = 'ref', speakers = speakers) for i in range(len(decoded))]
		print('Alignment time: {:.02f} sec'.format(time.time() - tic_alignment))
	 	
		if args.max_segment_duration:
			ref_transcript, hyp_transcript = [list(sorted(sum(segments, []), key = transcripts.sort_key)) for segments in [ref_segments, hyp_segments]]
			if ref:
				ref_segments = list(transcripts.segment(ref_transcript, args.max_segment_duration))
				hyp_segments = list(transcripts.segment(hyp_transcript, ref_segments))
			else:
				hyp_segments = list(transcripts.segment(hyp_transcript, args.max_segment_duration))
				ref_segments = [[] for _ in hyp_segments]

		transcript = [dict(audio_path = audio_path, ref = ref, hyp = hyp, speaker = transcripts.speaker(ref = ref_transcript, hyp = hyp_transcript), cer = metrics.cer(hyp, ref), words = metrics.align_words(hyp, ref)[-1], alignment = dict(ref = ref_transcript, hyp = hyp_transcript), **transcripts.summary(hyp_transcript)) for ref_transcript, hyp_transcript in zip(ref_segments, hyp_segments) for ref, hyp in [(transcripts.join(ref = ref_transcript), transcripts.join(hyp = hyp_transcript))]]
		filtered_transcript = list(transcripts.prune(transcript, align_boundary_words = args.align_boundary_words, cer = args.cer, duration = args.duration, gap = args.gap, unk = args.unk, num_speakers = args.num_speakers))
		json.dump(filtered_transcript, open(transcript_path, 'w'), ensure_ascii = False, sort_keys = True, indent = 2)

		print('Filtered segments:', len(filtered_transcript), 'out of', len(transcript))
		print(transcript_path)
		if args.html:
			vis.transcript(os.path.join(args.output_path, os.path.basename(audio_path) + '.html'), args.sample_rate, args.mono, transcript, filtered_transcript)
		if args.txt:
			open(os.path.join(args.output_path, os.path.basename(audio_path) + '.txt'), 'w').write(hyp)
		print('Done: {:.02f} sec\n'.format(time.time() - tic))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--verbose', action = 'store_true')
	parser.add_argument('--checkpoint', required = True)
	parser.add_argument('--model')
	parser.add_argument('--batch-time-padding-multiple', type = int, default = 128)
	parser.add_argument('--ext', default = ['wav', 'mp3', 'opus', 'm4a'])
	parser.add_argument('--skip-processed', action = 'store_true')
	parser.add_argument('--skip-file-longer-than-hours', type=float, help = 'skip files with duration more than specified hours')
	parser.add_argument('--input-path', '-i', nargs = '+')
	parser.add_argument('--output-path', '-o', default = 'data/transcribe')
	parser.add_argument('--device', default = 'cuda', choices = ['cpu', 'cuda'])
	parser.add_argument('--fp16', choices = ['O0', 'O1', 'O2', 'O3'], default = None)
	parser.add_argument('--num-workers', type = int, default = 0)
	parser.add_argument('--mono', action = 'store_true')
	parser.add_argument('--audio-backend', default = 'ffmpeg', choices = ['sox', 'ffmpeg'])
	parser.add_argument('--decoder', default = 'GreedyDecoder', choices = ['GreedyDecoder', 'BeamSearchDecoder'])
	parser.add_argument('--decoder-topk', type = int, default = 1)
	parser.add_argument('--beam-width', type = int, default = 5000)
	parser.add_argument('--beam-alpha', type = float, default = 0.3)
	parser.add_argument('--beam-beta', type = float, default = 1.0)
	parser.add_argument('--lm')
	parser.add_argument('--vad', type = int, choices = [0, 1, 2, 3], default = False, nargs = '?')
	parser.add_argument('--align', action = 'store_true')
	parser.add_argument('--window-size-dilate', type = float, default = 1.0)
	parser.add_argument('--max-segment-duration', type = float, default = 2)
	parser.add_argument('--cer', type = transcripts.number_tuple)
	parser.add_argument('--duration', type = transcripts.number_tuple)
	parser.add_argument('--num-speakers', type = transcripts.number_tuple)
	parser.add_argument('--gap', type = transcripts.number_tuple)
	parser.add_argument('--unk', type = transcripts.number_tuple)
	parser.add_argument('--align-boundary-words', action = 'store_true')
	parser.add_argument('--speakers', nargs = '*')
	parser.add_argument('--replace-blank-series', type = int, default = 8)
	parser.add_argument('--html', action = 'store_true')
	parser.add_argument('--txt', action = 'store_true', help = 'store whole transcript in txt format need for assessments')
	parser.add_argument('--mono', action='store_true')
	parser.add_argument('--shuffle', action='store_true')
	parser.add_argument('--max-ref-len', type=int, default=45000)
	args = parser.parse_args()
	args.vad = args.vad if isinstance(args.vad, int) else 3
	main(args)
