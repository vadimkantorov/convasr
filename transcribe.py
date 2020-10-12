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
import transcript_generators
import ctc
import transcripts
import vis
import utils
import tokenizers
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
		args.num_input_features, args.sample_rate, args.window_size, args.window_stride, args.window, eps = 1e-6
	)

	text_config = json.load(open(checkpoint['args']['text_config']))
	text_pipeline = language_processing.ProcessingPipeline.make(text_config, checkpoint['args']['text_pipelines'][0])

	model = getattr(models, args.model or checkpoint['args']['model'])(
		args.num_input_features, [text_pipeline.tokenizer.vocab_size],
		frontend = frontend,
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


def main(args):
	utils.enable_jit_fusion()

	assert args.output_json or args.output_html or args.output_txt or args.output_csv, \
		'at least one of the output formats must be provided'
	os.makedirs(args.output_path, exist_ok = True)
	data_paths = [
		p for f in args.input_path for p in ([os.path.join(f, g) for g in os.listdir(f)] if os.path.isdir(f) else [f])
		if os.path.isfile(p) and any(map(p.endswith, args.ext))
	] + [p for p in args.input_path if any(map(p.endswith, ['.json', '.json.gz']))]
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
	val_data_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size = None, collate_fn = val_dataset.collate_fn, num_workers = args.num_workers
	)
	csv_sep = dict(tab = '\t', comma = ',')[args.csv_sep]
	output_lines = []  # only used if args.output_csv is True

	oom_handler = utils.OomHandler(max_retries = args.oom_retries)
	for i, (meta, s, x, xlen, y, ylen) in enumerate(val_data_loader):
		print(f'Processing: {i}/{num_examples}')

		meta = [val_dataset.meta.get(m['example_id']) for m in meta]
		audio_path = meta[0]['audio_path']

		if x.numel() == 0:
			print(f'Skipping empty [{audio_path}].')
			continue

		begin = meta[0]['begin']
		end = meta[0]['end']
		audio_name = transcripts.audio_name(audio_path)

		try:
			tic = time.time()
			y, ylen = y.to(args.device), ylen.to(args.device)
			log_probs, logits, olen = model(x.squeeze(1).to(args.device), xlen.to(args.device))

			print('Input:', audio_name)
			print('Input time steps:', log_probs.shape[-1], '| target time steps:', y.shape[-1])
			print(
				'Time: audio {audio:.02f} sec | processing {processing:.02f} sec'.format(
					audio = sum(transcripts.compute_duration(t) for t in meta), processing = time.time() - tic
				)
			)

			ts = (x.shape[-1] / args.sample_rate) * torch.linspace(0, 1, steps = log_probs.shape[-1])
			ts = ts.unsqueeze(0).expand(x.shape[0], -1).to(log_probs.device)
			channel = [t['channel'] for t in meta]
			speaker = [t['speaker'] for t in meta]
			ref_segments = [[
				dict(
					channel = channel[i],
					begin = meta[i]['begin'],
					end = meta[i]['end'],
					ref = text_pipeline.postprocess(text_pipeline.preprocess(meta[i]['ref']))
				)
			] for i in range(len(meta))]
			##TODO add channel and speaker into segments
			hyp_segments = [_transcripts[0] for _transcripts in
			                generator.generate(text_pipeline.tokenizer,
			                                  log_probs,
			                                  torch.tensor([m['begin'] for m in meta], dtype = torch.float, device = 'cpu'),
			                                  torch.tensor([m['end'] for m in meta], dtype = torch.float, device = 'cpu'),
			                                  olen,
			                                  ts)]
			hyp_segments = transcripts.tag_segments(hyp_segments, 'hyp')

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
				aligned_ts = ts.gather(1, alignment)
				##TODO add channel and speaker into segments
				ref_segments = [_transcripts[0] for _transcripts in
				                generator.generate(text_pipeline.tokenizer,
				                                   torch.nn.functional.one_hot(y[:, 0, :], num_classes = log_probs.shape[1]).permute(0, 2, 1),
				                                   torch.tensor([m['begin'] for m in meta], dtype = torch.float, device = 'cpu'),
				                                   torch.tensor([m['end'] for m in meta], dtype = torch.float, device = 'cpu'),
				                                   ylen,
				                                   aligned_ts)]
				ref_segments = transcripts.tag_segments(ref_segments, 'ref')
			oom_handler.reset()
		except:
			if oom_handler.try_recover(model.parameters()):
				print(f'Skipping {i} / {num_examples}')
				continue
			else:
				raise

		print('Alignment time: {:.02f} sec'.format(time.time() - tic_alignment))

		if args.max_segment_duration:
			ref_transcript, hyp_transcript = [list(sorted(sum(segments, []), key = transcripts.sort_key)) for segments in [ref_segments, hyp_segments]]
			if ref:
				ref_segments = list(transcripts.segment(ref_transcript, args.max_segment_duration))
				hyp_segments = list(transcripts.segment(hyp_transcript, ref_segments))
			else:
				hyp_segments = list(transcripts.segment(hyp_transcript, args.max_segment_duration))
				ref_segments = [[] for _ in hyp_segments]

		transcript = [
			dict(
				audio_path = audio_path,
				ref = ref,
				hyp = hyp,
				speaker = transcripts.speaker(ref = ref_transcript, hyp = hyp_transcript),
				cer = metrics.cer(hyp = hyp, ref = ref),
				words = metrics.align_words(hyp = hyp, ref = ref)[-1] if args.align_words else [],
				alignment = dict(ref = ref_transcript, hyp = hyp_transcript),
				**transcripts.summary(hyp_transcript)
			) for ref_transcript,
			hyp_transcript in zip(ref_segments, hyp_segments) for ref,
			hyp in [(transcripts.join(ref = ref_transcript), transcripts.join(hyp = hyp_transcript))]
		]
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
			with open(transcript_path, 'w') as f:
				json.dump(filtered_transcript, f, ensure_ascii = False, sort_keys = True, indent = 2)

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
	parser.add_argument('--vad', type = int, choices = [0, 1, 2, 3], default = False, nargs = '?')
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
	args = parser.parse_args()
	args.vad = args.vad if isinstance(args.vad, int) else 3
	main(args)
