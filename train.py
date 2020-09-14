import logging
import logging.handlers
import argparse
import json
import math
import os
import random
import shutil
import time
import multiprocessing
import torch.utils.data
import torch.utils.tensorboard
import onnxruntime
import apex
import datasets
import decoders
#import exphtml
import metrics
import models
import optimizers
import torch
import transforms
import vis
import utils
import transcripts
import perf

class JsonlistSink:
	def __init__(self, log_path):
		self.log = open(log_path, 'w') if log_path else None

	def perf(self, perf, iteration, train_dataset):
		if self.log is None:
			return

		self.log.write(json.dumps(dict(train_dataset_name = train_dataset_name, loss_avg = perf['avg_loss'], lr_avg = perf['avg_lr'], iteration = iteration)))
		self.log.write('\n')

class TensorboardSink:
	def __init__(self, summary_writer):
		self.summary_writer = summary_writer

	def perf(self, perf, iteration, train_dataset_name, lr_scaler = 1e4):
		self.summary_writer.add_scalars(f'datasets/{train_dataset_name}', dict(loss_avg = perf['avg_loss'], lr_avg_scaled = perf['avg_lr'] * lr_scaler), iteration)
		#TODO: was add_scalar before, do not dump everything, or filter by prefix
		self.summary_writer.add_scalars('perf', {k.replace('datasets_', 'datasets/') : v for k, v in perf.items()}, iteration)

	def val_stats(self, iteration, val_dataset_name, labels_name, perf):
		prefix = f'datasets/{val_dataset_name}_{labels_name}_'
		self.summary_writer.add_scalars(
			dict(
				wer = perf[prefix + 'wer'] * 100.0,
				cer = perf[prefix + 'cer'] * 100.0,
				loss = perf[perfix + 'loss']
			),
			iteration
		)

		# https://github.com/pytorch/pytorch/issues/24234
		self.summary_writer.flush()

	def weight_stats(self, iteration, model, log_weight_distribution, eps = 1e-9):
		for param_name, param in models.master_module(model).named_parameters():
			if param.grad is None:
				continue

			tag = 'params/' + param_name.replace('.', '/')
			norm, grad_norm = param.norm(), param.grad.norm()
			ratio = grad_norm / (eps + norm)
			self.summary_writer.add_scalars(
				tag,
				dict(norm = float(norm), grad_norm = float(grad_norm), ratio = float(ratio)),
				iteration
			)
			
			if log_weight_distribution:
				self.summary_writer.add_histogram(tag, param, iteration)
				self.summary_writer.add_histogram(tag + '/grad', param.grad, iteration)
	

def apply_model(data_loader, model, labels, decoder, device, oom_handler):
	for meta, s, x, xlen, y, ylen in data_loader:
		x, xlen, y, ylen = x.to(device, non_blocking = True), xlen.to(device, non_blocking = True), y.to(device, non_blocking = True), ylen.to(device, non_blocking = True)
		with torch.no_grad():
			try:
				logits, log_probs, olen, loss = map(model(x, xlen, y = y, ylen = ylen).get, ['logits', 'log_probs', 'olen', 'loss'])
				oom_handler.reset()
			except:
				if not oom_handler.try_recover(model.parameters(), _print = utils.get_root_logger_logging_print()):
					raise

		entropy_char, *entropy_bpe = list(map(models.entropy, log_probs, olen))
		hyp = [list(map(l.decode, d.decode(lp, o))) for l, d, lp, o in zip(labels, decoder, log_probs, olen)]
		logits = list(map(models.unpad, logits, olen))
		y = list(map(models.unpad, y, ylen))
		yield meta, loss.cpu(), entropy_char.cpu(), hyp, logits, y


def evaluate_model(
	args,
	val_data_loaders,
	model,
	labels,
	decoder,
	error_analyzer,
	optimizer = None,
	sampler = None,
	tensorboard = None,
	tensorboard_sink = None,
	logfile_sink = None,
	epoch = None,
	iteration = None
):
	logging_print = utils.get_root_logger_logging_print()

	training = epoch is not None and iteration is not None
	columns = {}
	for val_dataset_name, val_data_loader in val_data_loaders.items():
		logging_print(f'\n{val_dataset_name}@{iteration}')
		transcript, logits_, y_ = [], [], []
		analyze = args.analyze == [] or (args.analyze is not None and val_dataset_name in args.analyze)

		model.eval()
		if args.adapt_bn:
			models.reset_bn_running_stats_(model)
			for _ in apply_model(val_data_loader, model, labels, decoder, args.device, args.val_crash_oom, utils.OomHandler(retries = args.oom_retries)):
				pass
		model.eval()
		cpu_list = lambda l: [[t.cpu() for t in t_] for t_ in l]

		tic = time.time()
		ref_, hyp_, audio_path_, loss_, entropy_ = [], [], [], [], []
		for batch_idx, (meta, loss, entropy, hyp, logits, y) in enumerate(apply_model(
				val_data_loader, model, labels, decoder, args.device, args.oom_retries)):
			loss_.extend(loss.tolist())
			entropy_.extend(entropy.tolist())
			logits_.extend(zip(*cpu_list(logits)) if not training and args.logits else [])
			y_.extend(cpu_list(y))
			audio_path_.extend(m['audio_path'] for m in meta)
			ref_.extend(labels[0].normalize_text(m['ref']) for m in meta)
			hyp_.extend(zip(*hyp))
		toc_apply_model = time.time()
		time_sec_val_apply_model = toc_apply_model - tic
		perf.update(dict(time_sec_val_apply_model = time_sec_val_apply_model), prefix = f'datasets_{val_dataset_name}')
		logging_print(f"Apply model {time_sec_val_apply_model:.1f} sec", end=", ")

		analyze_args_gen = (
			(
				hyp,
				ref,
				analyze,
				dict(
					labels_name=label.name,
					audio_path=audio_path,
					audio_name=transcripts.audio_name(audio_path),
					loss=loss,
					entropy=entropy
				),
				label.postprocess_transcript,
			)
			for ref, hyp_tuple, audio_path, loss, entropy in zip(ref_, hyp_, audio_path_, loss_, entropy_)
			for label, hyp in zip(labels, hyp_tuple)
		)

		if args.analyze_num_workers <= 0:
			transcript = [error_analyzer.analyze(*args) for args in analyze_args_gen]
		else:
			with multiprocessing.pool.Pool(processes = args.analyze_num_workers) as pool:
				transcript = pool.starmap(error_analyzer.analyze, analyze_args_gen)

		toc_analyze = time.time()
		time_sec_analyze = toc_analyze - toc_apply_model
		time_sec_total = toc_analyze - tic
		perf.update(dict(time_sec_val_analyze = time_sec_val_analyze, time_sec_val_total = time_sec_val_total), prefix = f'datasets_{val_dataset_name}')
		logging_print(f"Analyze {time_sec_val_analyze:.1f} sec, Total {time_sec_val_total:.1f} sec")
		
		for i, t in enumerate(transcript if args.verbose else []):		
			logging_print(f'{val_dataset_name}@{iteration}: {i // len(labels)} / {len(audio_path_)} | {args.experiment_id}')
			# TODO: don't forget to fix aligned hyp & ref output!
			# hyp = new_transcript['alignment']['hyp'] if analyze else new_transcript['hyp']
			# ref = new_transcript['alignment']['ref'] if analyze else new_transcript['ref']
			logging_print('REF: {labels_name} "{ref}"'.format(**t))
			logging_print('HYP: {labels_name} "{hyp}"'.format(**t))
			logging_print('WER: {labels_name} {wer:.02%} | CER: {cer:.02%}\n'.format(**t))

		transcripts_path = os.path.join(
			args.experiment_dir,
			args.train_transcripts_format
			.format(val_dataset_name = val_dataset_name, epoch = epoch, iteration = iteration)
		) if training else args.val_transcripts_format.format(
			val_dataset_name = val_dataset_name, decoder = args.decoder
		)
		for i, label in enumerate(labels):
			transcript_by_label = transcript[i:: len(labels)]
			aggregated = error_analyzer.aggregate(transcript_by_label)
			if analyze:
				with open(f'{transcripts_path}.errors.csv', 'w') as f:
					f.writelines('{hyp},{ref},{error_tag}\n'.format(**w) for w in aggregated['errors']['words'])

			logging_print('errors', aggregated['errors']['distribution'])
			cer = torch.FloatTensor([r['cer'] for r in transcript_by_label])
			loss = torch.FloatTensor([r['loss'] for r in transcript_by_label])
			logging_print('cer', metrics.quantiles(cer))
			logging_print('loss', metrics.quantiles(loss))
			logging_print(
				f'{args.experiment_id} {val_dataset_name} {label.name}',
				f'| epoch {epoch} iter {iteration}' if training else '',
				f'| {transcripts_path} |',
				('Entropy: {entropy:.02f} Loss: {loss:.02f} | WER:  {wer:.02%} CER: {cer:.02%} [{words_easy_errors_easy__cer_pseudo:.02%}],  MER: {mer_wordwise:.02%} DER: {hyp_der:.02%}/{ref_der:.02%}\n')
				.format(**aggregated)
			)
			#columns[val_dataset_name + '_' + labels_name] = {'cer' : aggregated['cer_avg'], '.wer' : aggregated['wer_avg'], '.loss' : aggregated['loss_avg'], '.entropy' : aggregated['entropy_avg'], '.cer_easy' : aggregated['cer_easy_avg'], '.cer_hard':  aggregated['cer_hard_avg'], '.cer_missing' : aggregated['cer_missing_avg'], 'E' : dict(value = aggregated['errors_distribution']), 'L' : dict(value = vis.histc_vega(loss, min = 0, max = 3, bins = 20), type = 'vega'), 'C' : dict(value = vis.histc_vega(cer, min = 0, max = 1, bins = 20), type = 'vega'), 'T' : dict(value = [('audio_name', 'cer', 'mer', 'alignment')] + [(r['audio_name'], r['cer'], r['mer'], vis.word_alignment(r['words'])) for r in sorted(r_, key = lambda r: r['mer'], reverse = True)] if analyze else [], type = 'table')}
			
			if training:
				perf.update(dict(
						wer = aggregated['wer'],
						cer = aggregated['cer'],
						loss = aggregated['loss']
					), prefix = f'datasets_val_{val_dataset_name}_{labels.name}'
				)
				tensorboard_sink.val_stats(iteration, val_dataset_name, label.name, perf)

		with open(transcripts_path, 'w') as f:
			json.dump(transcript, f, ensure_ascii = False, indent = 2, sort_keys = True)
		if analyze:
			vis.errors([transcripts_path], audio = args.vis_errors_audio)

		# TODO: transcript got flattened make this code work:
		# if args.logits:
		# 	logits_file_path = args.logits.format(val_dataset_name = val_dataset_name)
		# 	torch.save(
		# 		list(
		# 			sorted([
		# 				dict(
		# 					**r_,
		# 					logits = l_ if not args.logits_topk else models.sparse_topk(l_, args.logits_topk, dim = 0),
		# 					y = y_,
		# 					ydecoded = labels_.decode(y_.tolist())
		# 				) for r,
		# 				l,
		# 				y in zip(transcript, logits_, y_) for labels_,
		# 				r_,
		# 				l_,
		# 				y_ in zip(labels, r, l, y)
		# 			],
		# 					key = lambda r: r['cer'],
		# 					reverse = True)
		# 		),
		# 		logits_file_path
		# 	)
		# 	logging_print('Logits saved:', logits_file_path)

	checkpoint_path = os.path.join(
		args.experiment_dir, args.checkpoint_format.format(epoch = epoch, iteration = iteration)
	) if training and not args.checkpoint_skip else None
	#if args.exphtml:
	#	columns['checkpoint_path'] = checkpoint_path
	#	exphtml.expjson(args.exphtml, args.experiment_id, epoch = epoch, iteration = iteration, meta = vars(args), columns = columns, tag = 'train' if training else 'test', git_http = args.githttp)
	#	exphtml.exphtml(args.exphtml)

	if training and not args.checkpoint_skip:
		torch.save(
			dict(
				model_state_dict = models.master_module(model).state_dict(),
				optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None,
				amp_state_dict = apex.amp.state_dict() if args.fp16 else None,
				sampler_state_dict = sampler.state_dict() if sampler is not None else None,
				epoch = epoch,
				iteration = iteration,
				args = vars(args),
				time = time.time(),
				labels = [(l.name, str(l)) for l in labels]
			),
			checkpoint_path
		)

	model.train()


def main(args):
	checkpoints = [torch.load(checkpoint_path, map_location = 'cpu') for checkpoint_path in args.checkpoint]
	checkpoint = (checkpoints + [{}])[0]
	if len(checkpoints) > 1:
		checkpoint['model_state_dict'] = {
			k: sum(c['model_state_dict'][k] for c in checkpoints) / len(checkpoints)
			for k in checkpoint['model_state_dict']
		}

	if args.frontend_checkpoint:
		frontend_checkpoint = torch.load(args.frontend_checkpoint, map_location = 'cpu')
		frontend_extra_args = frontend_checkpoint['args']
		frontend_checkpoint = frontend_checkpoint['model']
	else:
		frontend_extra_args = None
		frontend_checkpoint = None

	args.experiment_id = args.experiment_id.format(
		model = args.model,
		frontend = args.frontend,
		train_batch_size = args.train_batch_size,
		optimizer = args.optimizer,
		lr = args.lr,
		weight_decay = args.weight_decay,
		time = time.strftime('%Y-%m-%d_%H-%M-%S'),
		experiment_name = args.experiment_name,
		bpe = 'bpe' if args.bpe else '',
		train_waveform_transform = f'aug{args.train_waveform_transform[0]}{args.train_waveform_transform_prob or ""}'
		if args.train_waveform_transform else '',
		train_feature_transform = f'aug{args.train_feature_transform[0]}{args.train_feature_transform_prob or ""}'
		if args.train_feature_transform else ''
	).replace('e-0', 'e-').rstrip('_')
	if checkpoint and 'experiment_id' in checkpoint['args'] and not args.experiment_name:
		args.experiment_id = checkpoint['args']['experiment_id']
	args.experiment_dir = args.experiment_dir.format(
		experiments_dir = args.experiments_dir, experiment_id = args.experiment_id
	)

	utils.set_up_root_logger(os.path.join(args.experiment_dir, 'log.txt'), mode = 'w')
	logging_print = utils.get_root_logger_logging_print()
	
	if checkpoint:
		args.lang, args.model, args.num_input_features, args.sample_rate, args.window, args.window_size, args.window_stride = map(checkpoint['args'].get, ['lang', 'model', 'num_input_features', 'sample_rate', 'window', 'window_size', 'window_stride'])

	logging_print('\n', 'Arguments:', args)
	logging_print('\n', 'Experiment id:', args.experiment_id, '\n')
	if args.dry:
		return
	utils.set_random_seed(args.seed)
	if args.cudnn == 'benchmark':
		torch.backends.cudnn.benchmark = True

	lang = datasets.Language(args.lang)
	#TODO: , candidate_sep = datasets.Labels.candidate_sep
	normalize_text_config = json.load(open(args.normalize_text_config)) if os.path.exists(args.normalize_text_config
																							) else {}
	labels = [datasets.Labels(lang, name = 'char', normalize_text_config = normalize_text_config)] + [
		datasets.Labels(lang, bpe = bpe, name = f'bpe{i}', normalize_text_config = normalize_text_config) for i,
		bpe in enumerate(args.bpe)
	]
	frontend = getattr(models, args.frontend)(
		out_channels = args.num_input_features,
		sample_rate = args.sample_rate,
		window_size = args.window_size,
		window_stride = args.window_stride,
		window = args.window,
		dither = args.dither,
		dither0 = args.dither0,
		stft_mode = 'conv' if args.onnx else None,
		extra_args = frontend_extra_args
	)
	model = getattr(models, args.model)(
		num_input_features = args.num_input_features,
		num_classes = list(map(len, labels)),
		dropout = args.dropout,
		decoder_type = 'bpe' if args.bpe else None,
		frontend = frontend if args.onnx or args.frontend_in_model else None,
		**(dict(inplace = False, dict = lambda logits, log_probs, olen, **kwargs: logits[0]) if args.onnx else {})
	)

	logging_print(' Model capacity:', int(models.compute_capacity(model, scale = 1e6)), 'million parameters\n')

	if checkpoint:
		model.load_state_dict(checkpoint['model_state_dict'], strict = False)

	if frontend_checkpoint:
		frontend_checkpoint = {
			'model.' + name: weight
			for name, weight in frontend_checkpoint.items()
		}  ##TODO remove after save checkpoint naming fix
		frontend.load_state_dict(frontend_checkpoint)

	if args.onnx:
		torch.set_grad_enabled(False)
		model.eval()
		model.to(args.device)
		model.fuse_conv_bn_eval()

		if args.fp16:
			model = models.InputOutputTypeCast(model.to(torch.float16), dtype = torch.float16)

		waveform_input = torch.rand(args.onnx_sample_batch_size, args.onnx_sample_time, device = args.device)
		logits = model(waveform_input)

		torch.onnx.export(
			model, (waveform_input, ),
			args.onnx,
			opset_version = args.onnx_opset,
			export_params = args.onnx_export_params,
			do_constant_folding = True,
			input_names = ['x'],
			output_names = ['logits'],
			dynamic_axes = dict(x = {
				0: 'B', 1: 'T'
			}, logits = {
				0: 'B', 2: 't'
			})
		)
		onnxruntime_session = onnxruntime.InferenceSession(args.onnx)
		if args.verbose:
			onnxruntime.set_default_logger_severity(0)
		(logits_, ) = onnxruntime_session.run(None, dict(x = waveform_input.cpu().numpy()))
		assert torch.allclose(logits.cpu(), torch.from_numpy(logits_), rtol = 1e-02, atol = 1e-03)
		
		#model_def = onnx.load(args.onnx)
		#import onnx.tools.net_drawer # import GetPydotGraph, GetOpNodeProducer
		#pydot_graph = GetPydotGraph(model_def.graph, name=model_def.graph.name, rankdir="TB", node_producer=GetOpNodeProducer("docstring", color="yellow", fillcolor="yellow", style="filled"))
		#pydot_graph.write_dot("pipeline_transpose2x.dot")
		#os.system('dot -O -Gdpi=300 -Tpng pipeline_transpose2x.dot')
		# add metadata to model
		return

	val_config = json.load(open(args.val_config)) if os.path.exists(args.val_config) else {}
	word_tags = json.load(open(args.word_tags)) if os.path.exists(args.word_tags) else {}
	for word_tag, words in val_config.get('word_tags', {}).items():
		word_tags[word_tag] = word_tags.get(word_tag, []) + words
	vocab = set(map(str.strip, open(args.vocab))) if os.path.exists(args.vocab) else set()
	error_analyzer = metrics.ErrorAnalyzer(metrics.WordTagger(lang, vocab = vocab, word_tags = word_tags), metrics.ErrorTagger(), val_config.get('error_analyzer', {}))

	make_transform = lambda name_args, prob: None if not name_args else getattr(transforms, name_args[0])(*name_args[1:]) if prob is None else getattr(transforms, name_args[0])(prob, *name_args[1:]) if prob > 0 else None
	val_frontend = models.AugmentationFrontend(
		frontend,
		waveform_transform = make_transform(args.val_waveform_transform, args.val_waveform_transform_prob),
		feature_transform = make_transform(args.val_feature_transform, args.val_feature_transform_prob)
	)

	if args.val_waveform_transform_debug_dir:
		args.val_waveform_transform_debug_dir = os.path.join(
			args.val_waveform_transform_debug_dir,
			str(val_frontend.waveform_transform)
			if isinstance(val_frontend.waveform_transform, transforms.RandomCompose) else
			val.waveform_transform.__class__.__name__
		)
		os.makedirs(args.val_waveform_transform_debug_dir, exist_ok = True)

	val_data_loaders = {
		os.path.basename(val_data_path): torch.utils.data.DataLoader(
			val_dataset,
			num_workers = args.num_workers,
			collate_fn = val_dataset.collate_fn,
			pin_memory = True,
			shuffle = False,
			batch_size = args.val_batch_size,
			worker_init_fn = datasets.worker_init_fn,
			timeout = args.timeout if args.num_workers > 0 else 0
		)
		for val_data_path in args.val_data_path for val_dataset in [
			datasets.AudioTextDataset(
				val_data_path,
				labels,
				args.sample_rate,
				frontend = val_frontend if not args.frontend_in_model else None,
				waveform_transform_debug_dir = args.val_waveform_transform_debug_dir,
				min_duration = args.min_duration,
				time_padding_multiple = args.batch_time_padding_multiple,
				pop_meta = True
			)
		]
	}
	decoder = [
		decoders.GreedyDecoder() if args.decoder == 'GreedyDecoder' else decoders.BeamSearchDecoder(
			labels[0],
			lm_path = args.lm,
			beam_width = args.beam_width,
			beam_alpha = args.beam_alpha,
			beam_beta = args.beam_beta,
			num_workers = args.num_workers,
			topk = args.decoder_topk
		)
	] + [decoders.GreedyDecoder() for bpe in args.bpe]

	model.to(args.device)

	if not args.train_data_path:
		model.eval()
		if not args.adapt_bn:
			model.fuse_conv_bn_eval()
		if args.device != 'cpu':
			model, *_ = models.data_parallel_and_autocast(model, opt_level = args.fp16, keep_batchnorm_fp32 = args.fp16_keep_batchnorm_fp32)
		evaluate_model(args, val_data_loaders, model, labels, decoder, error_analyzer)
		return

	model.freeze(backbone = args.freeze_backbone, decoder0 = args.freeze_decoder, frontend = args.freeze_frontend)

	train_frontend = models.AugmentationFrontend(
		frontend,
		waveform_transform = make_transform(args.train_waveform_transform, args.train_waveform_transform_prob),
		feature_transform = make_transform(args.train_feature_transform, args.train_feature_transform_prob)
	)
	tic = time.time()
	train_dataset = datasets.AudioTextDataset(
		args.train_data_path,
		labels,
		args.sample_rate,
		frontend = train_frontend if not args.frontend_in_model else None,
		min_duration = args.min_duration,
		max_duration = args.max_duration,
		time_padding_multiple = args.batch_time_padding_multiple,
		bucket = lambda example: int(
			math.ceil(
				((example[0]['end'] - example[0]['begin']) / args.window_stride + 1) / args.batch_time_padding_multiple
			)
		),
		pop_meta = True
	)

	logging_print('Time train dataset created:', time.time() - tic, 'sec')
	train_dataset_name = '_'.join(map(os.path.basename, args.train_data_path))
	tic = time.time()
	sampler = datasets.BucketingBatchSampler(
		train_dataset,
		batch_size = args.train_batch_size,
	)
	logging_print('Time train sampler created:', time.time() - tic, 'sec')

	train_data_loader = torch.utils.data.DataLoader(
		train_dataset,
		num_workers = args.num_workers,
		collate_fn = train_dataset.collate_fn,
		pin_memory = True,
		batch_sampler = sampler,
		worker_init_fn = datasets.worker_init_fn,
		timeout = args.timeout if args.num_workers > 0 else 0
	)
	optimizer = torch.optim.SGD(
		model.parameters(),
		lr = args.lr,
		momentum = args.momentum,
		weight_decay = args.weight_decay,
		nesterov = args.nesterov
	) if args.optimizer == 'SGD' else torch.optim.AdamW(
		model.parameters(), lr = args.lr, betas = args.betas, weight_decay = args.weight_decay
	) if args.optimizer == 'AdamW' else optimizers.NovoGrad(
		model.parameters(), lr = args.lr, betas = args.betas, weight_decay = args.weight_decay
	) if args.optimizer == 'NovoGrad' else apex.optimizers.FusedNovoGrad(
		model.parameters(), lr = args.lr, betas = args.betas, weight_decay = args.weight_decay
	) if args.optimizer == 'FusedNovoGrad' else None

	if checkpoint and checkpoint['optimizer_state_dict'] is not None:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		if not args.skip_optimizer_reset:
			optimizers.reset_options(optimizer)

	scheduler = optimizers.MultiStepLR(optimizer, gamma = args.decay_gamma, milestones = args.decay_milestones
										) if args.scheduler == 'MultiStepLR' else optimizers.PolynomialDecayLR(
											optimizer,
											power = args.decay_power,
											decay_steps = len(train_data_loader) * args.decay_epochs,
											end_lr = args.decay_lr
										) if args.scheduler == 'PolynomialDecayLR' else optimizers.NoopLR(optimizer)
	epoch, iteration = 0, 0
	if checkpoint:
		epoch, iteration = checkpoint['epoch'], checkpoint['iteration']
		if args.train_data_path == checkpoint['args']['train_data_path']:
			sampler.load_state_dict(checkpoint['sampler_state_dict'])
			if args.iterations_per_epoch and iteration and iteration % args.iterations_per_epoch == 0:
				sampler.batch_idx = 0
				epoch += 1
		else:
			epoch += 1
	if args.iterations_per_epoch:
		epoch_skip_fraction = 1 - args.iterations_per_epoch / len(train_data_loader)
		assert epoch_skip_fraction < args.max_epoch_skip_fraction, \
			f'args.iterations_per_epoch must not skip more than {args.max_epoch_skip_fraction:.1%} of each epoch'

	if args.device != 'cpu':
		model, optimizer = models.data_parallel_and_autocast(model, optimizer, opt_level = args.fp16, keep_batchnorm_fp32 = args.fp16_keep_batchnorm_fp32)
	if checkpoint and args.fp16 and checkpoint['amp_state_dict'] is not None:
		apex.amp.load_state_dict(checkpoint['amp_state_dict'])

	model.train()

	os.makedirs(args.experiment_dir, exist_ok = True)
	tensorboard_dir = os.path.join(args.experiment_dir, 'tensorboard')
	if checkpoint and args.experiment_name:
		tensorboard_dir_checkpoint = os.path.join(os.path.dirname(args.checkpoint[0]), 'tensorboard')
		if os.path.exists(tensorboard_dir_checkpoint) and not os.path.exists(tensorboard_dir):
			shutil.copytree(tensorboard_dir_checkpoint, tensorboard_dir)
	tensorboard = torch.utils.tensorboard.SummaryWriter(tensorboard_dir)
	tensorboard_sink = TensorboardSink(tensorboard)
	if args.log_json:
		args.log_json = os.path.join(args.experiment_dir, 'log.json')
	logfile_sink = JsonlistSink(args.log_json)


	perf.default.__init__(loss = dict(K = 50, max = 1000), memory_cuda_allocated = dict(K = 50), entropy = dict(K = 4), time_ms_iteration = dict(K = 50, max = 10_000), lr = dict(K = 50, max = 1))
	
	with open(os.path.join(args.experiment_dir, args.args), 'w') as f:
		json.dump(vars(args), f, sort_keys = True, ensure_ascii = False, indent = 2)

	with open(os.path.join(args.experiment_dir, args.dump_model_config), 'w') as f:
		model_config = dict(init_params = models.master_module(model).init_params, model = repr(models.master_module(model)))
		json.dump(model_config, f, sort_keys = True, ensure_ascii = False, indent = 2)

	tic, toc_fwd, toc_bwd = time.time(), time.time(), time.time()

	oom_hanlder = utils.OomHandler(max_retries = args.oom_retries)
	for epoch in range(epoch, args.epochs):
		sampler.shuffle(epoch + args.seed_sampler)
		time_epoch_start = time.time()
		for batch_idx, (meta, s, x, xlen, y, ylen) in enumerate(train_data_loader, start = sampler.batch_idx):
			toc_data = time.time()
			if batch_idx == 0:
				time_ms_launch_data_loader = (toc_data - tic) * 1000
				logging_print('Time data loader launch @ ', epoch, ':', time_ms_launch_data_loader / 1000, 'sec')
			
			lr = optimizer.param_groups[0]['lr']
			perf.update(dict(lr = lr))

			x, xlen, y, ylen = x.to(args.device, non_blocking = True), xlen.to(args.device, non_blocking = True), y.to(args.device, non_blocking = True), ylen.to(args.device, non_blocking = True)
			try:
				#TODO check nan values in tensors, they can break running_stats in bn
				log_probs, olen, loss = map(model(x, xlen, y = y, ylen = ylen).get, ['log_probs', 'olen', 'loss'])
				oom_handler.reset()
			except:
				if oom_handler.try_recover(model.parameters(), _print = logging_print):
					continue
				else:
					raise
			example_weights = ylen[:, 0]
			loss, loss_cur = (loss * example_weights).mean() / args.train_batch_accumulate_iterations, float(loss.mean())

			perf.update(dict(loss = loss_cur))

			entropy = float(models.entropy(log_probs[0], olen[0], dim = 1).mean())
			toc_fwd = time.time()
			#TODO: inf/nan still corrupts BN stats
			if not (torch.isinf(loss) or torch.isnan(loss)):
				if args.fp16:
					with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
						scaled_loss.backward()
				else:
					loss.backward()

				if iteration % args.train_batch_accumulate_iterations == 0:
					torch.nn.utils.clip_grad_norm_(
						apex.amp.master_params(optimizer) if args.fp16 else model.parameters(), args.max_norm
					)
					optimizer.step()

					if iteration > 0 and iteration % args.log_iteration_interval == 0:
						perf.update(utils.compute_memory_stats())
						tensorboard_sink.perf(perf.default, iteration, train_dataset_name)
						tensorboard_sink.weight_stats(iteration, model, args.log_weight_distribution)
						logfile_sink.perf(perf.default, iteration, train_dataset_name)

					optimizer.zero_grad()
					scheduler.step(iteration)
				perf.update(dict(entropy = entropy))
			toc_bwd = time.time()

			time_ms_data, time_ms_fwd, time_ms_bwd, time_ms_model = map(lambda sec: sec * 1000, [toc_data - tic, toc_fwd - toc_data, toc_bwd - toc_fwd, toc_bwd - toc_data])
			perf.update(dict(time_ms_data = time_ms_data, time_ms_fwd = time_ms_fwd, time_ms_bwd = time_ms_bwd, time_ms_iteration = time_ms_data + time_ms_model))

			print_left = f'{args.experiment_id} | epoch: {epoch:02d} iter: [{batch_idx: >6d} / {len(train_data_loader)} {iteration: >6d}] {"x".join(map(str, x.shape))}'
			print_right = 'ent: <{avg_entropy:.2f}> loss: {cur_loss:.2f} <{avg_loss:.2f}> time: {cur_time_ms_data:.2f}+{cur_time_ms_fwd:4.0f}+{cur_time_ms_bwd:4.0f} <{avg_time_ms_iteration:.0f}> | lr: {cur_lr:.5f}'.format(**perf.default)
			logging_print(print_left, print_right)
			iteration += 1
			sampler.batch_idx += 1

			if iteration > 0 and (iteration % args.val_iteration_interval == 0 or iteration == args.iterations):
				evaluate_model(
					args,
					val_data_loaders,
					model,
					labels,
					decoder,
					error_analyzer,
					optimizer,
					sampler,
					tensorboard,
					tensorboard_sink,
					epoch,
					iteration
				)

			if iteration and args.iterations and iteration >= args.iterations:
				return

			if args.iterations_per_epoch and iteration > 0 and iteration % args.iterations_per_epoch == 0:
				break

			tic = time.time()

		sampler.batch_idx = 0
		logging_print('Epoch time', (time.time() - time_epoch_start) / 60, 'minutes')
		if not args.skip_on_epoch_end_evaluation:
			evaluate_model(
				args,
				val_data_loaders,
				model,
				labels,
				decoder,
				error_analyzer,
				optimizer,
				sampler,
				tensorboard,
				tensorboard_sink,
				logfile_sink,
				epoch + 1,
				iteration
			)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--optimizer', choices = ['SGD', 'AdamW', 'NovoGrad', 'FusedNovoGrad'], default = 'SGD')
	parser.add_argument('--max-norm', type = float, default = 100)
	parser.add_argument('--lr', type = float, default = 1e-2)
	parser.add_argument('--skip-optimizer-reset', action = 'store_true')
	parser.add_argument('--weight-decay', type = float, default = 1e-3)
	parser.add_argument('--momentum', type = float, default = 0.9)
	parser.add_argument('--nesterov', action = 'store_true')
	parser.add_argument('--betas', nargs = '*', type = float, default = (0.9, 0.999))
	parser.add_argument('--scheduler', choices = ['MultiStepLR', 'PolynomialDecayLR'], default = None)
	parser.add_argument('--decay-gamma', type = float, default = 0.1)
	parser.add_argument(
		'--decay-milestones', nargs = '*', type = int, default = [25_000, 50_000], help = 'for MultiStepLR'
	)
	parser.add_argument('--decay-power', type = float, default = 2.0, help = 'for PolynomialDecayLR')
	parser.add_argument('--decay-lr', type = float, default = 1e-5, help = 'for PolynomialDecayLR')
	parser.add_argument('--decay-epochs', type = int, default = 5, help = 'for PolynomialDecayLR')
	parser.add_argument('--decay-step-size', type = int, default = 10_000, help = 'for PolynomialDecayLR')
	parser.add_argument('--fp16', choices = ['O0', 'O1', 'O2', 'O3'], default = None)
	parser.add_argument('--fp16-keep-batchnorm-fp32', default = None, action = 'store_true')
	parser.add_argument('--epochs', type = int, default = 5)
	parser.add_argument('--iterations', type = int, default = None)
	parser.add_argument('--iterations-per-epoch', type = int, default = None)
	parser.add_argument(
		'--max-epoch-skip-fraction',
		type = float,
		default = 0.20,
		help = 'limitation on how many samples will be skipped from each epoch, using some --iterations-per-epoch value'
	)
	parser.add_argument('--train-data-path', nargs = '*', default = [])
	parser.add_argument('--train-data-mixing', type = float, nargs = '*')
	parser.add_argument('--val-data-path', nargs = '*', default = [])
	parser.add_argument('--num-workers', type = int, default = 64)
	parser.add_argument('--train-batch-size', type = int, default = 256)
	parser.add_argument('--val-batch-size', type = int, default = 256)
	parser.add_argument('--device', default = 'cuda', choices = ['cuda', 'cpu'])
	parser.add_argument(
		'--checkpoint',
		nargs = '*',
		default = [],
		help = 'simple checkpoint weight averaging, for advanced weight averaging see Stochastic Weight Averaging'
	)
	parser.add_argument('--frontend-checkpoint', help = 'fariseq wav2vec checkpoint for frontend initialization')
	parser.add_argument('--checkpoint-skip', action = 'store_true')
	parser.add_argument('--skip-on-epoch-end-evaluation', action = 'store_true')
	parser.add_argument('--experiments-dir', default = 'data/experiments')
	parser.add_argument('--experiment-dir', default = '{experiments_dir}/{experiment_id}')
	parser.add_argument('--checkpoint-format', default = 'checkpoint_epoch{epoch:02d}_iter{iteration:07d}.pt')
	parser.add_argument(
		'--val-transcripts-format',
		default = 'data/transcripts_{val_dataset_name}_{decoder}.json',
		help = 'save transcripts at validation'
	)
	parser.add_argument(
		'--train-transcripts-format',
		default = 'transcripts_{val_dataset_name}_epoch{epoch:02d}_iter{iteration:07d}.json'
	)
	parser.add_argument(
		'--logits', nargs = '?', const = 'data/logits_{val_dataset_name}.pt', help = 'save logits at validation'
	)
	parser.add_argument('--logits-topk', type = int)
	parser.add_argument('--args', default = 'args.json', help = 'save experiment arguments to the experiment dir')
	parser.add_argument(
		'--dump-model-config', default = 'model.json', help = 'save model configuration to the experiment dir'
	)
	parser.add_argument('--model', default = 'JasperNetBig')
	parser.add_argument('--frontend', default = 'LogFilterBankFrontend')
	parser.add_argument(
		'--seed',
		type = int,
		default = 1,
		help = 'reset to this random seed value in the main thread and data loader threads'
	)
	parser.add_argument(
		'--seed-sampler',
		type = int,
		default = 0,
		help = 'this value is used as summand to random seed value in data sampler'
	)
	parser.add_argument('--cudnn', default = 'benchmark', help = 'enables cudnn benchmark mode')
	parser.add_argument(
		'--experiment-id',
		default =
		'{model}_{optimizer}_lr{lr:.0e}_wd{weight_decay:.0e}_bs{train_batch_size}_{train_waveform_transform}_{train_feature_transform}_{bpe}_{experiment_name}'
	)
	parser.add_argument('--experiment-name', '--name', default = '')
	parser.add_argument('--comment', default = '')
	parser.add_argument('--dry', action = 'store_true')
	parser.add_argument('--lang', default = 'ru')
	parser.add_argument('--val-waveform-transform-debug-dir', help = 'debug dir for augmented audio at validation')
	parser.add_argument(
		'--val-waveform-transform',
		nargs = '*',
		default = [],
		help = 'waveform transforms are applied before frontend to raw audio waveform'
	)
	parser.add_argument(
		'--val-waveform-transform-prob', type = float, default = None, help = 'apply transform with given probability'
	)
	parser.add_argument(
		'--val-feature-transform',
		nargs = '*',
		default = [],
		help = 'feature aug transforms are applied after frontend'
	)
	parser.add_argument('--val-feature-transform-prob', type = float, default = None)
	parser.add_argument('--train-waveform-transform', nargs = '*', default = [])
	parser.add_argument('--train-waveform-transform-prob', type = float, default = None)
	parser.add_argument('--train-feature-transform', nargs = '*', default = [])
	parser.add_argument('--train-feature-transform-prob', type = float, default = None)
	parser.add_argument(
		'--train-batch-accumulate-iterations', type = int, default = 1, help = 'number of gradient accumulation steps'
	)
	parser.add_argument('--val-iteration-interval', type = int, default = 2500)
	parser.add_argument('--log-iteration-interval', type = int, default = 100)
	parser.add_argument('--log-weight-distribution', action = 'store_true')
	parser.add_argument('--verbose', action = 'store_true', help = 'print all transcripts to console, at validation')
	parser.add_argument('--analyze', nargs = '*', default = None)
	parser.add_argument(
		'--decoder',
		default = 'GreedyDecoder',
		choices = ['GreedyDecoder', 'GreedyNoBlankDecoder', 'BeamSearchDecoder']
	)
	parser.add_argument(
		'--decoder-topk', type = int, default = 1, help = 'compute CER for many decoding hypothesis (oracle)'
	)
	parser.add_argument('--beam-width', type = int, default = 500)
	parser.add_argument(
		'--beam-alpha',
		type = float,
		default = 0.4,
		help = 'weight for language model (prob? log-prob?, TODO: check in ctcdecode)'
	)
	parser.add_argument(
		'--beam-beta',
		type = float,
		default = 2.6,
		help = 'weight for decoded transcript length (TODO: check in ctcdecode)'
	)
	parser.add_argument('--lm', help = 'path to KenLM language model for ctcdecode')
	parser.add_argument('--bpe', nargs = '*', help = 'path to SentencePiece subword model', default = [])
	parser.add_argument(
		'--timeout', type = float, default = 0, help = 'crash after specified timeout spent in DataLoader'
	)
	parser.add_argument(
		'--max-duration',
		type = float,
		default = 10,
		help = 'in seconds; drop in DataLoader all utterances longer than this value'
	)
	parser.add_argument(
		'--min-duration',
		type = float,
		default = 0.1,
		help = 'in seconds; drop in DataLoader all utterances smaller than this value'
	)
	parser.add_argument('--exphtml', default = '../stt_results')
	parser.add_argument('--freeze-backbone', type = int, default = 0, help = 'freeze number of backbone layers')
	parser.add_argument('--freeze-decoder', action = 'store_true', help = 'freeze decoder0')
	parser.add_argument('--freeze-frontend', action = 'store_true', help = 'freeze frontend')
	parser.add_argument('--num-input-features', type = int, default = 64, help = 'num of features produced by frontend')
	parser.add_argument('--sample-rate', type = int, default = 8_000, help = 'for frontend')
	parser.add_argument('--window-size', type = float, default = 0.02, help = 'for frontend, in seconds')
	parser.add_argument('--window-stride', type = float, default = 0.01, help = 'for frontend, in seconds')
	parser.add_argument('--dither0', type = float, default = 0.0, help = 'Amount of dithering prior to preemph')
	parser.add_argument('--dither', type = float, default = 1e-5, help = 'Amount of dithering after preemph')
	parser.add_argument(
		'--window', default = 'hann_window', choices = ['hann_window', 'hamming_window'], help = 'for frontend'
	)
	parser.add_argument('--onnx')
	parser.add_argument('--onnx-sample-batch-size', type = int, default = 16)
	parser.add_argument('--onnx-sample-time', type = int, default = 1024)
	parser.add_argument('--onnx-opset', type = int, default = 12, choices = [9, 10, 11, 12])
	parser.add_argument('--onnx-export-params', type = bool, default = True)
	parser.add_argument('--dropout', type = float, default = 0.2)
	parser.add_argument('--githttp')
	parser.add_argument('--vis-errors-audio', action = 'store_true')
	parser.add_argument('--adapt-bn', action = 'store_true')
	parser.add_argument('--vocab', default = 'data/vocab_word_list.txt')
	parser.add_argument('--word-tags', default = 'data/word_tags.json')
	parser.add_argument('--normalize-text-config', default = 'data/normalize_text_config.json')
	parser.add_argument('--frontend-in-model', action = 'store_true')
	parser.add_argument('--batch-time-padding-multiple', type = int, default = 128)
	parser.add_argument('--oom-retries', type = int, default = 3)
	parser.add_argument('--val-config', default = 'configs/ru_val_config.json')
	parser.add_argument('--analyze-num-workers', type = int, default = 0)
	parser.add_argument('--log-json', action = 'store_true')
	
	args = parser.parse_args()

	try:
		main(parser.parse_args())
	except Exception as e:
		logging.getLogger().exception(e)
		raise
