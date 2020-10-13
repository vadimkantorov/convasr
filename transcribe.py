#TODO:
# batch by vad
# figure out stft shift
# upstream ctc changes
# gpu levenshtein, needleman/hirschberg

import os
import time
import json
import argparse
import torch
import datasets
import models
import metrics
import transcript_generators
import ctc
import transcripts
import vis
import utils
import shaping
import language_processing

def legacy_compatibility_fix(args: dict):
	if 'lang' in args and args['lang'] == 'ru':
		args['text_config'] = 'configs/ru_text_config.json'
		args['text_pipelines'] = ['char_legacy']
	else:
		raise RuntimeError('"args.lang" no supported more, manually add "text_config" and "text_pipelines" properties to your checkpoint')
	return args

def setup(args):
	torch.set_grad_enabled(False)
	checkpoint = torch.load(args.checkpoint, map_location = 'cpu')
	checkpoint['args'] = legacy_compatibility_fix(checkpoint['args'])
	args.sample_rate, args.window_size, args.window_stride, args.window, args.num_input_features = map(checkpoint['args'].get, ['sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features'])
	frontend = models.LogFilterBankFrontend(
			args.num_input_features,
			args.sample_rate,
			args.window_size,
			args.window_stride,
			args.window,
			eps = 1e-6,
			normalize_signal=args.normalize_signal,
			debug_short_long_records_normalize_signal_multiplier=args.debug_short_long_records_normalize_signal_multiplier
	)

	text_config = json.load(open(checkpoint['args']['text_config']))
	text_pipeline = language_processing.ProcessingPipeline.make(text_config, checkpoint['args']['text_pipelines'][0])

	model = getattr(models, args.model or checkpoint['args']['model'])(
		args.num_input_features, [text_pipeline.tokenizer.vocab_size],
		frontend = frontend if args.frontend_in_model else None,
		dict = lambda logits,
		log_probs,
		olen,
		**kwargs: (log_probs[0], logits[0], olen[0])
	)
	model.load_state_dict(checkpoint['model_state_dict'], strict = False)
	model = model.to(args.device)
	model.eval()
	model.fuse_conv_bn_eval()
	if args.device != 'cpu':
		model, *_ = models.data_parallel_and_autocast(model, opt_level = args.fp16)
	generator = transcript_generators.GreedyCTCGenerator()
	return text_pipeline, frontend, model, generator


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

	text_pipeline, frontend, model, generator = setup(args)
	val_dataset = datasets.AudioTextDataset(
		data_paths, [text_pipeline],
		args.sample_rate,
		frontend = frontend if not args.frontend_in_model else None,
		segmented = True,
		mono = args.mono,
		time_padding_multiple = args.batch_time_padding_multiple,
		audio_backend = args.audio_backend,
		exclude = exclude,
		max_duration = args.transcribe_first_n_sec,
		join_transcript = args.join_transcript,
		string_array_encoding = args.dataset_string_array_encoding,
		debug_short_long_records_features_from_whole_normalized_signal = args.debug_short_long_records_features_from_whole_normalized_signal
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
		meta = [val_meta[t['example_id']] for t in meta]

		audio_path = meta[0]['audio_path']
		begin_end = [dict(begin = t['begin'], end = t['end']) for t in meta]
		audio_name = transcripts.audio_name(audio_path)
		#TODO check logic
		duration = x.shape[-1] / args.sample_rate
		channel = [t['channel'] for t in meta]
		speaker = [t.get('speaker', transcripts.speaker_missing) for t in meta]

		if x.numel() == 0:
			print(f'Skipping empty [{audio_path}].')
			continue

		try:
			tic = time.time()
			y, ylen = y.to(args.device), ylen.to(args.device)
			log_probs, logits, olen = model(x.squeeze(1).to(args.device), xlen.to(args.device))

			print('Input:', audio_name)
			print('Input time steps:', log_probs.shape[-1], '| target time steps:', y.shape[-1])
			print(
				'Time: audio {audio:.02f} sec | processing {processing:.02f} sec'.format(
					audio = sum(map(transcripts.compute_duration, meta)), processing = time.time() - tic
				)
			)

			ts = duration * torch.linspace(0, 1, steps = log_probs.shape[-1])
			ts = ts.unsqueeze(0).expand(x.shape[0], -1).to(log_probs.device)

			ref_segments = [[
				dict(
					channel = channel[i],
					begin = begin_end[i]['begin'],
					end = begin_end[i]['end'],
					ref = text_pipeline.postprocess(text_pipeline.preprocess(meta[i]['ref']))
				)
			] for i in range(len(meta))]
			##TODO add channel and speaker into segments
			hyp_segments = [alternatives[0] for alternatives in
			                generator.generate(tokenizer = text_pipeline.tokenizer,
			                                   log_probs = log_probs,
			                                   begin = torch.tensor([t['begin'] for t in begin_end], dtype = torch.float, device = 'cpu'),
			                                   end = torch.tensor([t['end'] for t in begin_end], dtype = torch.float, device = 'cpu'),
			                                   output_lengths = olen,
			                                   time_stamps = ts,
			                                   segment_text_key = 'hyp')]
			#TODO call text_pipeline.postprocess for hyp texts
			ref, hyp = '\n'.join(transcripts.join(ref = r) for r in ref_segments).strip(), '\n'.join(transcripts.join(hyp = h) for h in hyp_segments).strip()
			if args.verbose:
				print('HYP:', hyp)
			print('CER: {cer:.02%}'.format(cer = metrics.cer(hyp=hyp, ref=ref)))

			tic_alignment = time.time()
			if args.align and y.numel() > 0:
				alignment = ctc.alignment(
					log_probs.permute(2, 0, 1),
					y[:,0,:], # assumed that 0 channel is char labels
					olen,
					ylen[:,0],
					blank = text_pipeline.tokenizer.eps_id,
					pack_backpointers = args.pack_backpointers
				)
				aligned_ts: shaping.Bt = ts.gather(1, alignment)
				## TODO add channel and speaker into segments
				## TODO call text_pipeline.postprocess for ref texts
				ref_segments = [alternatives[0] for alternatives in
				                generator.generate(tokenizer = text_pipeline.tokenizer,
				                                   log_probs = torch.nn.functional.one_hot(y[:, 0, :], num_classes = log_probs.shape[1]).permute(0, 2, 1),
				                                   begin = torch.tensor([t['begin'] for t in begin_end], dtype = torch.float, device = 'cpu'),
				                                   end = torch.tensor([t['end'] for t in begin_end], dtype = torch.float, device = 'cpu'),
				                                   output_lengths = ylen,
				                                   time_stamps = aligned_ts,
			                                       segment_text_key = 'ref')]
			oom_handler.reset()
		except:
			if oom_handler.try_recover(model.parameters()):
				print(f'Skipping {i} / {num_examples}')
				continue
			else:
				raise

		print('Alignment time: {:.02f} sec'.format(time.time() - tic_alignment))

		ref_transcript, hyp_transcript = [sorted(transcripts.flatten(segments), key = transcripts.sort_key) for segments in [ref_segments, hyp_segments]]

		if args.max_segment_duration:
			if ref:
				ref_segments = list(transcripts.segment_by_time(ref_transcript, args.max_segment_duration))
				hyp_segments = list(transcripts.segment_by_ref(hyp_transcript, ref_segments))
			else:
				hyp_segments = list(transcripts.segment_by_time(hyp_transcript, args.max_segment_duration))
				ref_segments = [[] for _ in hyp_segments]

		elif args.join_transcript:
			ref_segments = [[t] for t in sorted(transcripts.load(audio_path + '.json'), key = transcripts.sort_key)]
			hyp_segments = list(transcripts.segment_by_ref(hyp_transcript, ref_segments, set_speaker = True, soft = False))

		has_ref = bool(transcripts.join(ref = transcripts.flatten(ref_segments)))

		transcript = []
		for ref_transcript, hyp_transcript in zip(ref_segments, hyp_segments):
			ref = transcripts.join(ref = ref_transcript)
			hyp = transcripts.join(hyp = hyp_transcript)
			transcript.append(
				dict(
					audio_path = audio_path,
					ref = ref,
					hyp = hyp,
					speaker_name = transcripts.speaker_name(ref = ref_transcript, hyp = hyp_transcript),

					words = metrics.align_words(_hyp_ = hyp, _ref_ = ref)[-1] if args.align_words else [],
					words_ref = ref_transcript,
					words_hyp = hyp_transcript,

					**transcripts.summary(hyp_transcript),
					**(dict(cer = metrics.cer(hyp = hyp, ref = ref)) if has_ref else {})
				)
			)

		transcripts.collect_speaker_names(transcript, set_speaker = True)

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

		# if args.output_csv:
		# 	[output_lines.append(csv_sep.join((audio_path, h, str(meta[i]['begin']), str(meta[i]['end']))) + '\n')
		# 	 for i, h in enumerate(hyp.split('\n'))]

		if args.logits:
			logits_file_path = os.path.join(args.output_path, audio_name + '.pt')
			if args.logits_crop:
				begin_end = [dict(zip(['begin', 'end'], [t['begin'] + c / float(o) * (t['end'] - t['begin']) for c in args.logits_crop])) for o, t in zip(olen, begin_end)]
				logits_crop = [slice(*args.logits_crop) for o in olen]
			else:
				logits_crop = [slice(int(o)) for o in olen]

			# TODO: filter ref / hyp by channel?
			torch.save([dict(audio_path = audio_path, logits = l[..., logits_crop[i]], **begin_end[i], ref = ref, hyp = hyp ) for i, l in enumerate(logits.cpu())], logits_file_path)
			print(logits_file_path)

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
	parser.add_argument('--logits', action = 'store_true')
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
	parser.add_argument('--normalize-signal', action = 'store_true')
	parser.add_argument('--debug-short-long-records-normalize-signal-multiplier', action = 'store_true')
	parser.add_argument('--debug-short-long-records-features-from-whole-normalized-signal', action = 'store_true')
	parser.add_argument('--frontend', type=str, default='LogFilterBankFrontend')
	parser.add_argument('--frontend-in-model', type=lambda x: bool(int(x or 0)), nargs='?', const=True, default=True)
	parser.add_argument('--diarize', action = 'store_true')
	parser.add_argument('--logits-crop', type = int, nargs = 2, default = [])
	args = parser.parse_args()
	main(args)
