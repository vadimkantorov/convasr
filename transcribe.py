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
import text_processing

def setup(args):
	torch.set_grad_enabled(False)
	checkpoint = torch.load(args.checkpoint, map_location = 'cpu')
	args.sample_rate, args.window_size, args.window_stride, args.window, args.num_input_features = map(checkpoint['args'].get, ['sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features'])
	frontend = models.LogFilterBankFrontend(
			args.num_input_features,
			args.sample_rate,
			args.window_size,
			args.window_stride,
			args.window,
			dither = args.dither,
			dither0 = args.dither0,
			#eps = 1e-6,
			normalize_signal=args.normalize_signal,
			debug_short_long_records_normalize_signal_multiplier=args.debug_short_long_records_normalize_signal_multiplier
	)

	# for legacy compat
	text_config = json.load(open(checkpoint['args'].get('text_config', args.text_config)))
	text_pipeline = text_processing.ProcessingPipeline.make(text_config, checkpoint['args'].get('text_pipelines', args.text_pipelines)[0])

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
			if basename.endswith('.json')]) if args.skip_processed else None

	data_paths = [path for path in data_paths if exclude is None or os.path.basename(path) not in exclude]

	text_pipeline, frontend, model, generator = setup(args)
	val_dataset = datasets.AudioTextDataset(
		data_paths, [text_pipeline],
		args.sample_rate,
		frontend = frontend if not args.frontend_in_model else None,
		mono = args.mono,
		time_padding_multiple = args.batch_time_padding_multiple,
		audio_backend = args.audio_backend,
		exclude = exclude,
		max_duration = args.transcribe_first_n_sec,
		mode = 'batched_channels' if args.join_transcript else 'batched_transcript',
		string_array_encoding = args.dataset_string_array_encoding,
		debug_short_long_records_features_from_whole_normalized_signal = args.debug_short_long_records_features_from_whole_normalized_signal,
		duration_from_transcripts=args.join_transcript
	)
	print('Examples count: ', len(val_dataset))
	val_meta = val_dataset.pop_meta()
	val_data_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size = None, collate_fn = val_dataset.collate_fn, num_workers = args.num_workers
	)
	csv_sep = dict(tab = '\t', comma = ',')[args.csv_sep]
	csv_lines = []  # only used if args.output_csv is True

	oom_handler = utils.OomHandler(max_retries = args.oom_retries)
	for i, (meta, s, x, xlen, y, ylen) in enumerate(val_data_loader):
		print(f'Processing: {i}/{len(val_dataset)}')
		meta = [val_meta[t['example_id']] for t in meta]

		audio_path = meta[0]['audio_path']
		audio_name = transcripts.audio_name(audio_path)
		begin_end = [dict(begin = t['begin'], end = t['end']) for t in meta]
		begin = torch.tensor([t['begin'] for t in begin_end], dtype = torch.float)	
		end = torch.tensor([t['end'] for t in begin_end], dtype = torch.float)
		#TODO WARNING assumes frontend not in dataset
		if not args.frontend_in_model:
			print('\n' * 10 + 'WARNING\n' * 5)
			print('transcribe.py assumes frontend in model, in other case time alignment was incorrect')
			print('WARNING\n' * 5 + '\n')

		duration = x.shape[-1] / args.sample_rate
		channel = [t['channel'] for t in meta]
		speaker = [t['speaker'] for t in meta]
		speaker_name = [t['speaker_name'] for t in meta]

		if x.numel() == 0:
			print(f'Skipping empty [{audio_path}].')
			continue

		try:
			tic = time.time()
			y, ylen = y.to(args.device), ylen.to(args.device)
			print('Input:', audio_name)
			print('Xlen:', x.shape[-1] / args.sample_rate)
			print('ylen:', y.shape[-1])

			log_probs, logits, olen = model(x.squeeze(1).to(args.device), xlen.to(args.device))

			print('Input time steps:', log_probs.shape[-1], '| target time steps:', y.shape[-1])
			print(
				'Time: audio {audio:.02f} sec | processing {processing:.02f} sec'.format(
					audio = sum(map(transcripts.compute_duration, meta)), processing = time.time() - tic
				)
			)

			ts: shaping.Bt = duration * torch.linspace(0, 1, steps = log_probs.shape[-1], device = log_probs.device).unsqueeze(0).expand(x.shape[0], -1)

			ref_segments = [[
				dict(
					channel = channel[i],
					begin = begin_end[i]['begin'],
					end = begin_end[i]['end'],
					ref = text_pipeline.postprocess(text_pipeline.preprocess(meta[i]['ref']))
				)
			] for i in range(len(meta))]
			hyp_segments = [alternatives[0] for alternatives in
							generator.generate(tokenizer = text_pipeline.tokenizer,
											   log_probs = log_probs,
											   begin = begin,
											   end = end,
											   output_lengths = olen,
											   time_stamps = ts,
											   segment_text_key = 'hyp',
											   segment_extra_info = [dict(speaker = s, speaker_name = sn, channel = c) for s, sn, c in zip(speaker, speaker_name, channel)])]
			hyp_segments = [transcripts.map_text(text_pipeline.postprocess, hyp = hyp) for hyp in hyp_segments]
			hyp, ref = '\n'.join(transcripts.join(hyp = h) for h in hyp_segments).strip(), '\n'.join(transcripts.join(ref = r) for r in ref_segments).strip()
			if args.verbose:
				print('HYP:', hyp)
			print('CER: {cer:.02%}'.format(cer = metrics.cer(hyp = hyp, ref = ref)))

			tic_alignment = time.time()
			if args.align and y.numel() > 0:
				alignment: shaping.BY = ctc.alignment(
					log_probs.permute(2, 0, 1),
					y[:,0,:], # assumed that 0 channel is char labels
					olen,
					ylen[:,0],
					blank = text_pipeline.tokenizer.eps_id,
					pack_backpointers = args.pack_backpointers
				)
				aligned_ts: shaping.Bt = ts.gather(1, alignment)

				ref_segments = [alternatives[0] for alternatives in
								generator.generate(tokenizer = text_pipeline.tokenizer,
												   log_probs = torch.nn.functional.one_hot(y[:, 0, :], num_classes = log_probs.shape[1]).permute(0, 2, 1),
												   begin = begin,
												   end = end,
												   output_lengths = ylen,
												   time_stamps = aligned_ts,
												   segment_text_key = 'ref',
												   segment_extra_info = [dict(speaker = s, speaker_name = sn, channel = c) for s, sn, c in zip(speaker, speaker_name, channel)])]
				ref_segments = [transcripts.map_text(text_pipeline.postprocess, ref = ref) for ref in ref_segments]
			oom_handler.reset()
		except:
			if oom_handler.try_recover(model.parameters()):
				print(f'Skipping {i} / {len(val_dataset)}')
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

		#### HACK for diarization
		elif args.ref_transcript_path and args.join_transcript:
			audio_name_hack = audio_name.split('.')[0]
			#TODO: normalize ref field
			ref_segments = [[t] for t in sorted(transcripts.load(os.path.join(args.ref_transcript_path, audio_name_hack + '.json')), key = transcripts.sort_key)]
			hyp_segments = list(transcripts.segment_by_ref(hyp_transcript, ref_segments, set_speaker = True, soft = False))
		#### END OF HACK

		has_ref = bool(transcripts.join(ref = transcripts.flatten(ref_segments)))

		transcript = []
		for hyp_transcript, ref_transcript in zip(hyp_segments, ref_segments):
			hyp, ref = transcripts.join(hyp = hyp_transcript), transcripts.join(ref = ref_transcript)

			transcript.append(
				dict(
					audio_path = audio_path,
					ref = ref,
					hyp = hyp,
					speaker_name = transcripts.speaker_name(ref = ref_transcript, hyp = hyp_transcript),

					words = metrics.align_words(*metrics.align_strings(hyp = hyp, ref = ref)) if args.align_words else [],
					words_ref = ref_transcript if args.align_words else [],
					words_hyp = hyp_transcript if args.align_words else [],

					**transcripts.summary(hyp_transcript),
					**(dict(cer = metrics.cer(hyp = hyp, ref = ref)) if has_ref else {})
				)
			)

		transcripts.collect_speaker_names(transcript, set_speaker_data = True, num_speakers = 2)

		filtered_transcript = list(
			transcripts.prune(
				transcript,
				align_boundary_words = args.align_boundary_words,
				cer = args.prune_cer,
				duration = args.prune_duration,
				gap = args.prune_gap,
				allowed_unk_count = args.prune_unk,
				num_speakers = args.prune_num_speakers
			)
		)

		print('Filtered segments:', len(filtered_transcript), 'out of', len(transcript))

		if args.output_json:
			transcript_path = os.path.join(args.output_path, audio_name + '.json')
			print(transcripts.save(transcript_path, filtered_transcript))

		if args.output_html:
			transcript_path = os.path.join(args.output_path, audio_name + '.html')
			print(vis.transcript(transcript_path, args.sample_rate, args.mono, transcript, filtered_transcript))

		if args.output_txt:
			transcript_path = os.path.join(args.output_path, audio_name + '.txt')
			with open(transcript_path, 'w') as f:
				f.write(' '.join(t['hyp'].strip() for t in filtered_transcript))
			print(transcript_path)

		if args.output_csv:
			assert len({t['audio_path'] for t in filtered_transcript}) == 1
			audio_path = filtered_transcript[0]['audio_path']
			hyp = ' '.join(t['hyp'].strip() for t in filtered_transcript)
			begin = min(t['begin'] for t in filtered_transcript)
			end = max(t['end'] for t in filtered_transcript)
			csv_lines.append(csv_sep.join([audio_path, hyp, str(begin), str(end)]))

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
		transcript_path = os.path.join(args.output_path, 'transcripts.csv')
		with open(transcript_path, 'w') as f:
			f.write('\n'.join(csv_lines))
		print(transcript_path)


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
	parser.add_argument('--max-segment-duration', type = float, default = 0.0)
	parser.add_argument('--prune-cer', type = transcripts.number_tuple)
	parser.add_argument('--prune-duration', type = transcripts.number_tuple)
	parser.add_argument('--prune-num-speakers', type = transcripts.number_tuple)
	parser.add_argument('--prune-gap', type = transcripts.number_tuple)
	parser.add_argument('--prune-unk', type = transcripts.number_tuple)
	parser.add_argument('--speakers', nargs = '*')
	parser.add_argument('--replace-blank-series', type = int, default = 8)
	parser.add_argument('--transcribe-first-n-sec', type = int)
	parser.add_argument('--join-transcript', action = 'store_true')
	parser.add_argument('--pack-backpointers', action = 'store_true')
	parser.add_argument('--oom-retries', type = int, default = 5)
	parser.add_argument('--dataset-string-array-encoding', default = 'utf_32_le', choices = ['utf_16_le', 'utf_32_le'])
	parser.add_argument('--normalize-signal', action = 'store_true')
	parser.add_argument('--debug-short-long-records-normalize-signal-multiplier', action = 'store_true')
	parser.add_argument('--debug-short-long-records-features-from-whole-normalized-signal', action = 'store_true')
	parser.add_argument('--frontend', type=str, default='LogFilterBankFrontend')
	parser.add_argument('--frontend-in-model', type = utils.str2bool, nargs = '?', const = True, default = True)
	parser.add_argument('--logits-crop', type = int, nargs = 2, default = [])
	parser.add_argument('--text-config', default = 'configs/ru_text_config.json')
	parser.add_argument('--text-pipelines', nargs = '+', help = 'text processing pipelines (names should be defined in text-config)', default = ['char_legacy'])
	parser.add_argument('--ref-transcript-path')
	parser.add_argument('--dither0', type = float, default = 0.0, help = 'Amount of dithering prior to preemph')
	parser.add_argument('--dither', type = float, default = 0.0, help = '1e-5 used in training. Amount of dithering after preemph')
	args = parser.parse_args()
	main(args)
