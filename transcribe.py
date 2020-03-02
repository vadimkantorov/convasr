#TODO:

# batch by vad
# figure out stft shift
# disable repeat deduplication
# upstream ctc changes
# gpu levenshtein, needleman/hirschberg

import os
import io
import time
import json
import base64
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

@torch.no_grad()
def main(args):
	os.makedirs(args.output_path, exist_ok = True)
	
	checkpoint = torch.load(args.checkpoint, map_location = 'cpu')
	args.sample_rate, args.window_size, args.window_stride, args.window, args.num_input_features = map(checkpoint['args'].get, ['sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features'])
	frontend = models.LogFilterBankFrontend(args.num_input_features, args.sample_rate, args.window_size, args.window_stride, args.window)
	labels = datasets.Labels(importlib.import_module(checkpoint['args']['lang']), name = 'char')
	model = getattr(models, checkpoint['args']['model'])(args.num_input_features, [len(labels)], frontend = frontend, dict = lambda logits, log_probs, olen, **kwargs: (logits[0], olen[0]))
	model.load_state_dict(checkpoint['model_state_dict'], strict = False)
	model = model.to(args.device)
	model.eval()
	model.fuse_conv_bn_eval()
	#model = models.data_parallel(model)

	decoder = decoders.GreedyDecoder() if args.decoder == 'GreedyDecoder' else decoders.BeamSearchDecoder(labels, lm_path = args.lm, beam_width = args.beam_width, beam_alpha = args.beam_alpha, beam_beta = args.beam_beta, num_workers = args.num_workers, topk = args.decoder_topk)
	
	audio_paths = [p for f in args.data_path for p in ([os.path.join(f, g) for g in os.listdir(f)] if os.path.isdir(f) else [f]) if os.path.isfile(p) and any(map(p.endswith, args.ext))]

	val_dataset = datasets.AudioTextDataset(audio_paths, [labels], args.sample_rate, frontend = None, segmented = True, mono = args.mono, time_padding_multiple = args.batch_time_padding_multiple, vad_options = dict(window_size = args.window_size, aggressiveness = args.vad, window_size_dilate = args.window_size_dilate))
	val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size = None, collate_fn = val_dataset.collate_fn, num_workers = args.num_workers)

	for meta, x, xlen, y, ylen in val_data_loader:
		audio_path = meta[0]['audio_path']
		transcript_path = os.path.join(args.output_path, os.path.basename(audio_path) + '.json')

		tic = time.time()
		import IPython; IPython.embed()
		y, ylen = y.to(args.device), ylen.to(args.device)
		log_probs, olen = model(x.to(args.device), xlen.to(args.device))


		#speech = vad.detect_speech(x.squeeze(1), args.sample_rate, args.window_size, aggressiveness = args.vad, window_size_dilate = args.window_size_dilate)
		#speech = vad.upsample(speech, log_probs)
		#log_probs.masked_fill_(models.silence_space_mask(log_probs, speech, space_idx = labels.space_idx, blank_idx = labels.blank_idx), float('-inf'))
		
		decoded = decoder.decode(log_probs, olen)
		
		print(os.path.basename(audio_path))
		print('Input time steps:', log_probs.shape[-1], '| target time steps:', y.shape[-1])
		print('Time: audio {audio:.02f} sec | processing {processing:.02f} sec'.format(audio = sum(t['end'] - t['begin'] for t in meta), processing = time.time() - tic))

		ts = (x.shape[-1] / args.sample_rate) * torch.linspace(0, 1, steps = log_probs.shape[-1]).unsqueeze(0) + torch.FloatTensor([t['begin'] for t in meta]).unsqueeze(1)
		segments = [([dict(channel = meta[i]['channel'], begin = meta[i]['begin'], end = meta[i]['end'], ref = labels.decode(y[i, 0, :ylen[i]].tolist()))], labels.decode(decoded[i], ts[i], channel = meta[i]['channel'], replace_blank = True, replace_repeat = True, replace_space = False)) for i in range(len(decoded))]
		channel = [t['channel'] for t in meta]
		
		ref_full, y_full = labels.encode(meta[0]['ref_full'])

		ref, hyp = ' '.join(transcripts.join(ref = r) for r, h in segments).strip(), ' '.join(transcripts.join(hyp = h) for r, h in segments).strip()

		if args.verbose:
			print('HYP:', hyp)
		print('CER: {cer:.02%}'.format(cer = metrics.cer(hyp, ref or ref_full)))

		tic = time.time()
		if args.align:
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
			segments = [(labels.decode(y[i, 0, :ylen[i]].tolist(), ts[i], alignment[i], channel = channel[i], speaker = speaker[i], key = 'ref'), h) for i, (r, h) in enumerate(segments)]
		print('Alignment time: {:.02f} sec'.format(time.time() - tic))
		
		ref_segments, hyp_segments = zip(*segments)
		ref_segments, hyp_segments = sum(ref_segments, []), sum(hyp_segments, [])
		
		if args.max_segment_seconds:
			# can't resegment large txt because of slow needleman
			if ref_segments:
				ref_segments = list(transcripts.resegment(ref_segments, args.max_segment_seconds))
				hyp_segments = list(transcripts.resegment(hyp_segments, ref_segments))
			else:
				hyp_segments = transcripts.resegment(hyp_segments, args.max_segment_seconds)

		transcript = [dict(audio_path = audio_path, ref = ref_, hyp = hyp_, cer = metrics.cer(hyp_, ref_), words = metrics.align_words(hyp_, ref_)[-1], alignment = dict(ref = ref, hyp = hyp), **transcripts.summary(hyp)) for ref, hyp in zip(ref_segments, hyp_segments) for ref_, hyp_ in [(transcripts.join(ref = ref), transcripts.join(hyp = hyp))]]
		
		filtered_transcript = list(transcripts.filter(transcript, min_duration = args.min_duration, max_duration = args.max_duration, min_cer = args.min_cer, max_cer = args.max_cer, time_gap = args.gap, align_boundary_words = args.align_boundary_words))
		json.dump(transcripts.strip(filtered_transcript, args.strip), open(transcript_path, 'w'), ensure_ascii = False, sort_keys = True, indent = 2)
		print('Filtered segments:', len(filtered_transcript), 'out of', len(transcript))
		print(transcript_path)
		if args.html:
			print(vis.transcript(os.path.join(args.output_path, os.path.basename(audio_path) + '.html'), args.sample_rate, args.mono, transcript, filtered_transcript))	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', required = True)
	parser.add_argument('--data-path', '-i', nargs = '+')
	parser.add_argument('--output-path', '-o', default = 'data/transcribe')
	parser.add_argument('--device', default = 'cuda', choices = ['cpu', 'cuda'])
	parser.add_argument('--max-segment-seconds', type = float, default = 2)
	parser.add_argument('--num-workers', type = int, default = 0)
	parser.add_argument('--ext', default = ['wav', 'mp3'])
	parser.add_argument('--decoder', default = 'GreedyDecoder', choices = ['GreedyDecoder', 'BeamSearchDecoder'])
	parser.add_argument('--decoder-topk', type = int, default = 1)
	parser.add_argument('--beam-width', type = int, default = 5000)
	parser.add_argument('--beam-alpha', type = float, default = 0.3)
	parser.add_argument('--beam-beta', type = float, default = 1.0)
	parser.add_argument('--lm')
	parser.add_argument('--batch-time-padding-multiple', type = int, default = 128)
	parser.add_argument('--vad', type = int, choices = [0, 1, 2, 3], default = False, nargs = '?')
	parser.add_argument('--align', action = 'store_true')
	parser.add_argument('--verbose', action = 'store_true')
	parser.add_argument('--window-size-dilate', type = float, default = 1.0)
	parser.add_argument('--mono', action = 'store_true')
	parser.add_argument('--html', action = 'store_true')
	parser.add_argument('--min-cer', type = float)
	parser.add_argument('--max-cer', type = float)
	parser.add_argument('--min-duration', type = float)
	parser.add_argument('--max-duration', type = float)
	parser.add_argument('--gap', type = float)
	parser.add_argument('--align-boundary-words', action = 'store_true')
	parser.add_argument('--strip', nargs = '*', default = ['alignment'])
	args = parser.parse_args()
	args.vad = args.vad if isinstance(args.vad, int) else 3
	main(args)
