#TODO:
# batch by vad
# figure out stft shift
# upstream ctc changes
# gpu levenshtein, needleman/hirschberg

import gc
import os
import time
import json
import argparse
import torch
import datasets
import models
import metrics
import decoders
import ctc
import transcripts
import vis
import utils
import diarization


def setup(args):
	torch.set_grad_enabled(False)
	checkpoint = torch.load(args.checkpoint, map_location = 'cpu')
	args.sample_rate, args.window_size, args.window_stride, args.window, args.num_input_features = map(checkpoint['args'].get, ['sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features'])
	frontend = models.LogFilterBankFrontend(
		args.num_input_features, args.sample_rate, args.window_size, args.window_stride, args.window, eps = 1e-6
	)
	labels = datasets.Labels(datasets.Language(checkpoint['args']['lang']), name = 'char')
	model = getattr(models, args.model or checkpoint['args']['model'])(
		args.num_input_features, [len(labels)],
		frontend = frontend,
		dict = lambda logits,
		log_probs,
		olen,
		**kwargs: (logits[0], olen[0])
	)
	model.load_state_dict(checkpoint['model_state_dict'], strict = False)
	model = model.to(args.device)
	model.eval()
	model.fuse_conv_bn_eval()
	if args.device != 'cpu':
		model, *_ = models.data_parallel_and_autocast(model, opt_level = args.fp16)
	decoder = decoders.GreedyDecoder() if args.decoder == 'GreedyDecoder' else decoders.BeamSearchDecoder(
		labels,
		lm_path = args.lm,
		beam_width = args.beam_width,
		beam_alpha = args.beam_alpha,
		beam_beta = args.beam_beta,
		num_workers = args.num_workers,
		topk = args.decoder_topk
	)
	segmentation_model = diarization.PyannoteDiarizationModel() if args.diarize else diarization.WebrtcSpeechActivityDetectionModel() if args.vad is not False else None

	return labels, frontend, model, decoder, segmentation_model


def main(args, ext_json = ['.json', '.json.gz']):
	utils.enable_jit_fusion()

	assert args.output_json or args.output_html or args.output_txt or args.output_csv, \
		'at least one of the output formats must be provided'
	os.makedirs(args.output_path, exist_ok = True)
	
	audio_data_paths = set(
		p for f in args.input_path for p in ([os.path.join(f, g) for g in os.listdir(f)] if os.path.isdir(f) else [f])
		if os.path.isfile(p) and any(map(p.endswith, args.ext))
	)
	json_data_paths = set(p for p in args.input_path if any(map(p.endswith, ext_json)) and not utils.strip_suffixes(p, ext_json) in audio_data_paths)

	data_paths = list(audio_data_paths | json_data_paths)

	exclude = set(
		[os.path.splitext(basename)[0]
			for basename in os.listdir(args.output_path)
			if basename.endswith('.json')] if args.skip_processed else []
	)
	data_paths = [path for path in data_paths if os.path.basename(path) not in exclude]

	labels, frontend, model, decoder, segmentation_model = setup(args)
	val_dataset = datasets.AudioTextDataset(
		data_paths, [labels],
		args.sample_rate,
		frontend = None,
		segmented = True,
		mono = args.mono,
		time_padding_multiple = args.batch_time_padding_multiple,
		audio_backend = args.audio_backend,
		exclude = exclude,
		max_duration = args.transcribe_first_n_sec,
		join_transcript = args.join_transcript,
		string_array_encoding = args.dataset_string_array_encoding
	)
	num_examples = len(val_dataset)
	print('Examples count: ', num_examples)
	val_meta = val_dataset.pop_meta()
	val_data_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size = None, collate_fn = val_dataset.collate_fn, num_workers = args.num_workers
	)
	csv_sep = dict(tab = '\t', comma = ',')[args.csv_sep]
	output_lines = []  # only used if args.output_csv is True

	oom_handler = utils.OomHandler(max_retries = args.oom_retries)
	for i, (meta, s, x, xlen, y, ylen) in enumerate(val_data_loader):
		print(f'Processing: {i}/{num_examples}')

		meta = [val_meta.get(m['example_id']) for m in meta]
		audio_path, begin, end = map(meta[0].get, ['audio_path', 'begin', 'end'])
		audio_name = transcripts.audio_name(audio_path)

		if x.numel() == 0:
			print(f'Skipping empty [{audio_path}].')
			continue

		try:
			tic = time.time()
			y, ylen = y.to(args.device), ylen.to(args.device)
			log_probs, olen = model(x.squeeze(1).to(args.device), xlen.to(args.device))

			decoded = decoder.decode(log_probs, olen)

			print('Input:', audio_name)
			print('Input time steps:', log_probs.shape[-1], '| target time steps:', y.shape[-1])
			print(
				'Time: audio {audio:.02f} sec | processing {processing:.02f} sec'.format(
					audio = sum(transcripts.compute_duration(t) for t in meta), processing = time.time() - tic
				)
			)

			ts = (x.shape[-1] / args.sample_rate) * torch.linspace(0, 1, steps = log_probs.shape[
				-1]).unsqueeze(0) + torch.FloatTensor([t['begin'] for t in meta]).unsqueeze(1)
			channel = [t['channel'] for t in meta]
			speaker = [t['speaker'] for t in meta]
			ref_segments = [[
				dict(
					channel = channel[i],
					begin = meta[i]['begin'],
					end = meta[i]['end'],
					ref = labels.decode(y[i, 0, :ylen[i]].tolist())
				)
			] for i in range(len(decoded))]
			hyp_segments = [
				labels.decode(
					decoded[i],
					ts[i],
					channel = channel[i],
					replace_blank = True,
					replace_blank_series = args.replace_blank_series,
					replace_repeat = True,
					replace_space = False,
					speaker = speaker[i] if isinstance(speaker[i], str) else None
				) for i in range(len(decoded))
			]

			ref, hyp = '\n'.join(transcripts.join(ref = r) for r in ref_segments).strip(), '\n'.join(transcripts.join(hyp = h) for h in hyp_segments).strip()
			if args.verbose:
				print('HYP:', hyp)
			print('CER: {cer:.02%}'.format(cer = metrics.cer(hyp=hyp, ref=ref)))

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

				alignment = ctc.alignment(
					log_probs.permute(2, 0, 1),
					y.squeeze(1),
					olen,
					ylen.squeeze(1),
					blank = labels.blank_idx,
					pack_backpointers = args.pack_backpointers
				)
				ref_segments = [
					labels.decode(
						y[i, 0, :ylen[i]].tolist(),
						ts[i],
						alignment[i],
						channel = channel[i],
						speaker = speaker[i],
						key = 'ref',
						speakers = val_dataset.speakers
					) for i in range(len(decoded))
				]
			oom_handler.reset()
		except:
			if oom_handler.try_recover(model.parameters()):
				print(f'Skipping {i} / {num_examples}')
				continue
			else:
				raise

		print('Alignment time: {:.02f} sec'.format(time.time() - tic_alignment))

		ref_transcript, hyp_transcript = [list(sorted(sum(segments, []), key = transcripts.sort_key)) for segments in [ref_segments, hyp_segments]]

		if args.max_segment_duration:
			if ref:
				ref_segments = list(transcripts.segment_by_time(ref_transcript, args.max_segment_duration))
				hyp_segments = list(transcripts.segment_by_segments(hyp_transcript, ref_segments))
			else:
				hyp_segments = list(transcripts.segment_by_time(hyp_transcript, args.max_segment_duration))
				ref_segments = [[] for _ in hyp_segments]

		elif args.join_transcript:
			ref_segments = [[t] for t in transcripts.load(audio_path + '.json')]
			hyp_segments = list(transcripts.segment_by_segments(hyp_transcript, ref_segments, set_speaker = True))

		has_ref = any(t.get('ref') for ref_transcript in ref_segments for t in ref_transcript)
		transcript = [
			dict(
				audio_path = audio_path,
				ref = ref,
				hyp = hyp,
				speaker_name = transcripts.speaker_name(ref = ref_transcript, hyp = hyp_transcript),
				
				words = metrics.align_words(hyp = hyp, ref = ref)[-1] if args.align_words else [],
				words_ref = ref_transcript,
				words_hyp = hyp_transcript,
				
				**transcripts.summary(hyp_transcript),
				**(dict(cer = metrics.cer(hyp = hyp, ref = ref)) if has_ref else {})

			) for ref_transcript, hyp_transcript in zip(ref_segments, hyp_segments) for ref, hyp in [(transcripts.join(ref = ref_transcript), transcripts.join(hyp = hyp_transcript))]
		]
		transcripts.set_speaker(transcript)

		filtered_transcript = list(
			transcripts.prune(
				transcript,
				align_boundary_words = args.align_boundary_words,
				cer = args.cer,
				duration = args.duration,
				gap = args.gap,
				unk = args.unk,
				num_speakers = args.num_speakers
			)
		)

		print('Filtered segments:', len(filtered_transcript), 'out of', len(transcript))

		if args.output_json:
			transcript_path = os.path.join(args.output_path, audio_name + '.json')
			print(transcript_path)
			transcripts.save(transcript_path, filtered_transcript)

		if args.output_html:
			transcript_path = os.path.join(args.output_path, audio_name + '.html')
			print(transcript_path)
			vis.transcript(transcript_path, args.sample_rate, args.mono, transcript, filtered_transcript)

		if args.output_txt:
			transcript_path = os.path.join(args.output_path, audio_name + '.txt')
			print(transcript_path)
			with open(transcript_path, 'w') as f:
				f.write(hyp)

		if args.output_csv:
			output_lines.append(csv_sep.join((audio_path, hyp, str(begin), str(end))) + '\n')

		print('Done: {:.02f} sec\n'.format(time.time() - tic))

	if args.output_csv:
		with open(os.path.join(args.output_path, 'transcripts.csv'), 'w') as f:
			f.writelines(output_lines)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--verbose', action = 'store_true')
	parser.add_argument('--checkpoint', required = True)
	parser.add_argument('--model')
	parser.add_argument('--batch-time-padding-multiple', type = int, default = 128)
	parser.add_argument('--ext', default = ['wav', 'mp3', 'opus', 'm4a'])
	parser.add_argument('--skip-processed', action = 'store_true')
	parser.add_argument('--input-path', '-i', nargs = '+')
	parser.add_argument('--output-path', '-o', default = 'data/transcribe')
	parser.add_argument('--output-json', action = 'store_true', help = 'write transcripts to separate json files')
	parser.add_argument('--output-html', action = 'store_true', help = 'write transcripts to separate html files')
	parser.add_argument('--output-txt', action = 'store_true', help = 'write transcripts to separate txt files')
	parser.add_argument('--output-csv', action = 'store_true', help = 'write transcripts to a transcripts.csv file')
	parser.add_argument('--csv-sep', default = 'tab', choices = ['tab', 'comma'])
	parser.add_argument('--device', default = 'cuda', choices = ['cpu', 'cuda'])
	parser.add_argument('--fp16', choices = ['O0', 'O1', 'O2', 'O3'], default = None)
	parser.add_argument('--num-workers', type = int, default = 0)
	parser.add_argument('--mono', action = 'store_true')
	parser.add_argument('--audio-backend', default = None, choices = ['sox', 'ffmpeg'])
	parser.add_argument('--decoder', default = 'GreedyDecoder', choices = ['GreedyDecoder', 'BeamSearchDecoder'])
	parser.add_argument('--decoder-topk', type = int, default = 1)
	parser.add_argument('--beam-width', type = int, default = 5000)
	parser.add_argument('--beam-alpha', type = float, default = 0.3)
	parser.add_argument('--beam-beta', type = float, default = 1.0)
	parser.add_argument('--lm')
	parser.add_argument('--align', action = 'store_true')
	parser.add_argument('--align-boundary-words', action = 'store_true')
	parser.add_argument('--align-words', action = 'store_true')
	parser.add_argument('--window-size-dilate', type = float, default = 1.0)
	parser.add_argument('--max-segment-duration', type = float, default = 2)
	parser.add_argument('--cer', type = transcripts.number_tuple)
	parser.add_argument('--duration', type = transcripts.number_tuple)
	parser.add_argument('--num-speakers', type = transcripts.number_tuple)
	parser.add_argument('--gap', type = transcripts.number_tuple)
	parser.add_argument('--unk', type = transcripts.number_tuple)
	parser.add_argument('--speakers', nargs = '*')
	parser.add_argument('--replace-blank-series', type = int, default = 8)
	parser.add_argument('--transcribe-first-n-sec', type = int)
	parser.add_argument('--join-transcript', action = 'store_true')
	parser.add_argument('--pack-backpointers', action = 'store_true')
	parser.add_argument('--oom-retries', type = int, default = 3)
	parser.add_argument('--dataset-string-array-encoding', default = 'utf_32_le', choices = ['utf_16_le', 'utf_32_le'])
	parser.add_argument('--diarize', action = 'store_true')
	parser.add_argument('--vad', type = int, choices = [0, 1, 2, 3], default = False, nargs = '?')
	args = parser.parse_args()
	main(args)
