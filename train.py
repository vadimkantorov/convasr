#TODO: add optimization time logging
#TODO: add RSS logging
#TODO: add CPU/GPU load logging
#TODO: add PyTorch GPU caching allocator memory utilisation (utils.compute_memory_fragmentation) logging
#TODO: add data loader process amount logging
#TODO: add intra-op parallelism logging (including DataLoader)
#TODO: add validaation logging to json

import logging
import logging.handlers
import argparse
import json
import math
import os
import collections
import random
import shutil
import datetime
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
import itertools
import torch.distributed as dist

class JsonlistSink:
	def __init__(self, file_path, mode = 'w'):
		self.json_file = open(file_path, mode) if file_path else None

	def log(self, perf, iteration, train_dataset_name):
		if self.json_file is None:
			return

		self.json_file.write(json.dumps(dict(train_dataset_name = train_dataset_name,
											 loss_BT_normalized_avg = perf['avg_loss_BT_normalized'],
											 lr_avg = perf['avg_lr'],
											 time_ms_data_avg = perf['performance_avg_time_ms_data'],
											 time_ms_forward_avg = perf['performance_avg_time_ms_fwd'],
											 time_ms_backward_avg = perf['performance_avg_time_ms_bwd'],
											 input_B_cur = perf['performance_cur_input_B'],
											 input_T_cur = perf['performance_cur_input_T'],
											 iteration = iteration)))
		self.json_file.write('\n')
		self.json_file.flush()

class TensorboardSink:
	def __init__(self, summary_writer):
		self.summary_writer = summary_writer

	def perf(self, perf, iteration, rank = None):
		filter_key = 'performance_'
		aggregated_metrics = collections.defaultdict(dict)
		for key, value in perf.items():
			if filter_key == key[:len(filter_key)]:
				key = key[len(filter_key):]
			else:
				continue
			agg_type = key.split('_')[0]
			name = ''.join(key.split('_')[1:])
			aggregated_metrics[name][agg_type] = value

		for name, value in aggregated_metrics.items():
			if rank is not None:
				tag = f'perf/{name}_node<{rank}>'
			else:
				tag = f'perf/{name}'
			self.summary_writer.add_scalars(tag, value, iteration)

	def train_stats(self, perf, iteration, train_dataset_name, lr_scaler = 1e4):
		self.summary_writer.add_scalars(f'datasets/{train_dataset_name}', dict(loss_BT_normalized_avg=perf['avg_loss_BT_normalized'], lr_avg_scaled=perf['avg_lr'] * lr_scaler), iteration)

	def val_stats(self, iteration, val_dataset_name, labels_name, perf):
		prefix = f'datasets_val_{val_dataset_name}_{labels_name}_cur_'
		self.summary_writer.add_scalars(
			prefix.replace('datasets_', 'datasets/'),
			dict(
				wer = perf[prefix + 'wer'] * 100.0,
				cer = perf[prefix + 'cer'] * 100.0,
				loss = perf[prefix + 'loss']
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
				## TODO remove args.world_size > 1 after OOM recover fix for DDP
				if args.world_size > 1 or not oom_handler.try_recover(model.parameters(), _print = utils.get_root_logger_print(level = logging.ERROR)):
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
	tensorboard_sink = None,
	logfile_sink = None,
	epoch = None,
	iteration = None
):
	_print = utils.get_root_logger_print()

	training = epoch is not None and iteration is not None
	columns = {}
	oom_handler = utils.OomHandler(max_retries=args.oom_retries)
	for val_dataset_name, val_data_loader in val_data_loaders.items():
		if args.world_size > 1:
			# synchronize process, because rank 0 process do result analisys and io
			dist.barrier()
		_print(f'\n{val_dataset_name}@{iteration}')
		analyze = args.analyze == [] or (args.analyze is not None and val_dataset_name in args.analyze)

		model.eval()
		if args.adapt_bn:
			models.reset_bn_running_stats_(model)
			for _ in apply_model(val_data_loader, model, labels, decoder, args.device, oom_handler):
				pass
		model.eval()

		tic = time.time()
		ref_, hyp_, audio_path_, loss_, entropy_ = [], [], [], [], []
		for batch_idx, (meta, loss, entropy, hyp, logits, y) in enumerate(apply_model(
				val_data_loader, model, labels, decoder, args.device, oom_handler)):
			loss_.extend(loss.tolist())
			entropy_.extend(entropy.tolist())
			audio_path_.extend(m['audio_path'] for m in meta)
			ref_.extend(labels[0].normalize_text(m['ref']) for m in meta)
			hyp_.extend(zip(*hyp))
		toc_apply_model = time.time()
		time_sec_val_apply_model = toc_apply_model - tic
		perf.update(dict(time_sec_val_apply_model = time_sec_val_apply_model), prefix = f'datasets_{val_dataset_name}')
		_print(f"Apply model {time_sec_val_apply_model:.1f} sec")

		if args.world_size > 1:
			def sync_float_list(float_list):
				sync = utils.gather_tensors(torch.tensor(float_list, dtype=torch.float32, device=args.device), world_size=args.world_size)
				return list(itertools.chain.from_iterable([x.cpu().tolist() for x in sync]))

			def sync_string_list(str_list):
				string_storage = utils.TensorBackedStringArray(str_list, device = args.device)
				string_storage.synchronize(args.world_size)
				return list(string_storage.to('cpu'))
			_print(f"Synchronize validation results started")
			sync_tic = time.time()
			loss_ = sync_float_list(loss_)
			_print(f'loss sync time {time.time() - sync_tic:.1f} sec');sync_tic = time.time()
			entropy_ = sync_float_list(entropy_)
			_print(f'entropy sync time {time.time() - sync_tic:.1f} sec');sync_tic = time.time()
			audio_path_ = sync_string_list(audio_path_)
			_print(f'audio_path sync time {time.time()  - sync_tic:.1f} sec');sync_tic = time.time()
			ref_ = sync_string_list(ref_)
			_print(f'ref sync time {time.time() - sync_tic:.1f} sec');sync_tic = time.time()
			hyps_by_labels = list(map(list, zip(*hyp_)))
			for i in range(len(hyps_by_labels)):
				hyps_by_labels[i] = sync_string_list(hyps_by_labels[i])
			hyp_ = list(map(list, zip(*hyps_by_labels)))
			_print(f'hyp sync time {time.time()- sync_tic:.1f} sec')
			time_sec_val_sync_results = time.time() - toc_apply_model
			perf.update(dict(time_sec_val_sync_results=time_sec_val_sync_results), prefix=f'datasets_{val_dataset_name}')
			_print(f"Synchronize validation results {time_sec_val_sync_results:.1f} sec")
			if args.rank != 0:
				continue

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
		time_sec_val_analyze = toc_analyze - toc_apply_model
		time_sec_val_total = toc_analyze - tic
		perf.update(dict(time_sec_val_analyze = time_sec_val_analyze, time_sec_val_total = time_sec_val_total), prefix = f'datasets_{val_dataset_name}')
		_print(f"Analyze {time_sec_val_analyze:.1f} sec, Total {time_sec_val_total:.1f} sec")
		
		for i, t in enumerate(transcript if args.verbose else []):		
			_print(f'{val_dataset_name}@{iteration}: {i // len(labels)} / {len(audio_path_)} | {args.experiment_id}')
			# TODO: don't forget to fix aligned hyp & ref output!
			# hyp = new_transcript['alignment']['hyp'] if analyze else new_transcript['hyp']
			# ref = new_transcript['alignment']['ref'] if analyze else new_transcript['ref']
			_print('REF: {labels_name} "{ref}"'.format(**t))
			_print('HYP: {labels_name} "{hyp}"'.format(**t))
			_print('WER: {labels_name} {wer:.02%} | CER: {cer:.02%}\n'.format(**t))

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

			_print('errors', aggregated['errors']['distribution'])
			cer = torch.FloatTensor([r['cer'] for r in transcript_by_label])
			loss = torch.FloatTensor([r['loss'] for r in transcript_by_label])
			_print('cer', metrics.quantiles(cer))
			_print('loss', metrics.quantiles(loss))
			_print(
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
					), prefix = f'datasets_val_{val_dataset_name}_{label.name}'
				)
				tensorboard_sink.val_stats(iteration, val_dataset_name, label.name, perf.default())

		with open(transcripts_path, 'w') as f:
			json.dump(transcript, f, ensure_ascii = False, indent = 2, sort_keys = True)
		if analyze:
			vis.errors([transcripts_path], audio = args.vis_errors_audio)

	checkpoint_path = os.path.join(
		args.experiment_dir, args.checkpoint_format.format(epoch = epoch, iteration = iteration)
	) if training and not args.checkpoint_skip else None

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
		train_batch_size = args.train_batch_size * args.world_size,
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

	os.makedirs(args.experiment_dir, exist_ok = True)

	if checkpoint:
		args.lang, args.model, args.num_input_features, args.sample_rate, args.window, args.window_size, args.window_stride = map(checkpoint['args'].get, ['lang', 'model', 'num_input_features', 'sample_rate', 'window', 'window_size', 'window_stride'])
		log_mode = 'a'
	else:
		log_mode = 'w'

	if args.log_json:
		args.log_json = os.path.join(args.experiment_dir, f'log.node{args.rank}.json')

	log_path = os.path.join(args.experiment_dir, 'log.txt')
	if args.rank == 0:
		log_level = logging.INFO
	else:
		log_level = logging.ERROR

	utils.set_up_root_logger(log_path, mode = log_mode, level = log_level)
	logfile_sink = JsonlistSink(args.log_json, mode = log_mode)

	_print = utils.get_root_logger_print()
	_print('\n', 'Arguments:', args)
	_print(f'"CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", default = "")}"')
	_print(f'"CUDA_LAUNCH_BLOCKING={os.environ.get("CUDA_LAUNCH_BLOCKING", default="")}"')
	_print('Experiment id:', args.experiment_id, '\n')
	if args.dry:
		return
	if args.cudnn == 'benchmark':
		torch.backends.cudnn.benchmark = True

	lang = datasets.Language(args.lang)
	#TODO: , candidate_sep = datasets.Labels.candidate_sep
	normalize_text_config = json.load(open(args.normalize_text_config)) if os.path.exists(args.normalize_text_config) else {}
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

	_print('Model capacity:', int(models.compute_capacity(model, scale = 1e6)), 'million parameters\n')

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

	perf.init_default(loss=dict(K=50, max=1000), memory_cuda_allocated=dict(K=50), entropy=dict(K=4), time_ms_iteration=dict(K=50, max=10_000), lr=dict(K=50, max=1))

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
			val_frontend.waveform_transform.__class__.__name__
		)
		os.makedirs(args.val_waveform_transform_debug_dir, exist_ok = True)

	val_data_loaders = {}
	for val_data_path in args.val_data_path:
		val_dataset = datasets.AudioTextDataset(val_data_path, labels, args.sample_rate,
												frontend = val_frontend if not args.frontend_in_model else None,
												waveform_transform_debug_dir = args.val_waveform_transform_debug_dir,
												min_duration = args.min_duration,
												time_padding_multiple = args.batch_time_padding_multiple,
												pop_meta = True,
												_print = _print)
		if args.world_size > 1:
			sampler = torch.utils.data.DistributedSampler(val_dataset, num_replicas = args.world_size, rank = args.rank, shuffle = False)
		else:
			sampler = None

		val_data_loader = torch.utils.data.DataLoader(
			val_dataset,
			num_workers = args.num_workers,
			collate_fn = val_dataset.collate_fn,
			pin_memory = True,
			shuffle = False,
			batch_size = args.val_batch_size,
			worker_init_fn = datasets.worker_init_fn,
			sampler = sampler,
			timeout = args.timeout if args.num_workers > 0 else 0
		)

		val_data_loaders[os.path.basename(val_data_path)] = val_data_loader

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
		if args.world_size > 1:
			model, *_ = models.distributed_data_parallel_and_autocast(model, args.local_rank, opt_level = args.fp16, synchronize_bn = args.synchronize_bn, keep_batchnorm_fp32 = args.fp16_keep_batchnorm_fp32)
		elif args.device != 'cpu':
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
		pop_meta = True,
		_print = _print
	)

	_print('Time train dataset created:', time.time() - tic, 'sec')
	train_dataset_name = '_'.join(map(os.path.basename, args.train_data_path))
	tic = time.time()
	sampler = datasets.BucketingBatchSampler(
		train_dataset,
		batch_size = args.train_batch_size,
	)
	if args.world_size > 1:
		sampler = datasets.DistributedSamplerWrapper(sampler, num_replicas=args.world_size, rank=args.rank)
	_print('Time train sampler created:', time.time() - tic, 'sec')

	train_data_loader = torch.utils.data.DataLoader(
		train_dataset,
		num_workers = args.num_workers,
		collate_fn = train_dataset.collate_fn,
		pin_memory = True,
		batch_sampler = sampler,
		worker_init_fn = datasets.worker_init_fn,
		timeout = args.timeout if args.num_workers > 0 else 0
	)

	if args.optimizer == 'SGD':
		optimizer = torch.optim.SGD(model.parameters(),
									lr = args.lr,
									momentum = args.momentum,
									weight_decay = args.weight_decay,
									nesterov = args.nesterov)
	elif args.optimizer == 'AdamW':
		optimizer = torch.optim.AdamW(model.parameters(),
									   lr = args.lr,
									   betas = args.betas,
									   weight_decay = args.weight_decay)
	elif args.optimizer == 'NovoGrad':
		optimizer = optimizers.NovoGrad(model.parameters(),
										lr = args.lr,
										betas = args.betas,
										weight_decay = args.weight_decay)
	elif args.optimizer == 'FusedNovoGrad':
		optimizer = apex.optimizers.FusedNovoGrad(model.parameters(),
												  lr = args.lr,
												  betas = args.betas,
												  weight_decay = args.weight_decay)
	else:
		optimizer = None

	if checkpoint and checkpoint['optimizer_state_dict'] is not None:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		if not args.skip_optimizer_reset:
			optimizers.reset_options(optimizer)

	if args.scheduler == 'MultiStepLR':
		scheduler = optimizers.MultiStepLR(optimizer, gamma = args.decay_gamma, milestones = args.decay_milestones)
	elif args.scheduler == 'PolynomialDecayLR':
		scheduler = optimizers.PolynomialDecayLR(optimizer, power = args.decay_power, decay_steps = len(train_data_loader) * args.decay_epochs, end_lr = args.decay_lr)
	else:
		scheduler = optimizers.NoopLR(optimizer)

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

	if args.world_size > 1:
		model, optimizer = models.distributed_data_parallel_and_autocast(model, args.local_rank, optimizer, opt_level=args.fp16, keep_batchnorm_fp32=args.fp16_keep_batchnorm_fp32)
	elif args.device != 'cpu':
		model, optimizer = models.data_parallel_and_autocast(model, optimizer, opt_level=args.fp16, keep_batchnorm_fp32=args.fp16_keep_batchnorm_fp32)
	if checkpoint and args.fp16 and checkpoint['amp_state_dict'] is not None:
		apex.amp.load_state_dict(checkpoint['amp_state_dict'])

	model.train()

	tensorboard_dir = os.path.join(args.experiment_dir, 'tensorboard')
	utils.copy_tensorboard_dir_from_previous_checkpoint_if_exists(args, tensorboard_dir)

	if args.world_size > 1:
		if args.local_rank == 0:
			os.makedirs(tensorboard_dir, exist_ok=True)
		dist.barrier()
	tensorboard = torch.utils.tensorboard.SummaryWriter(tensorboard_dir)
	tensorboard_sink = TensorboardSink(tensorboard)

	if args.rank == 0:
		with open(os.path.join(args.experiment_dir, args.args), 'w') as f:
			json.dump(vars(args), f, sort_keys = True, ensure_ascii = False, indent = 2)

		with open(os.path.join(args.experiment_dir, args.dump_model_config), 'w') as f:
			model_config = dict(init_params = models.master_module(model).init_params, model = repr(models.master_module(model)))
			json.dump(model_config, f, sort_keys = True, ensure_ascii = False, indent = 2)

	tic, toc_fwd, toc_bwd = time.time(), time.time(), time.time()

	oom_handler = utils.OomHandler(max_retries = args.oom_retries)
	for epoch in range(epoch, args.epochs):
		sampler.set_epoch(epoch + args.seed_sampler)
		time_epoch_start = time.time()
		for batch_idx, (meta, s, x, xlen, y, ylen) in enumerate(train_data_loader, start = sampler.batch_idx):
			toc_data = time.time()
			if batch_idx == 0:
				time_ms_launch_data_loader = (toc_data - tic) * 1000
				_print('Time data loader launch @ ', epoch, ':', time_ms_launch_data_loader / 1000, 'sec')
			
			lr = optimizer.param_groups[0]['lr']
			perf.update(dict(lr = lr))

			x, xlen, y, ylen = x.to(args.device, non_blocking = True), xlen.to(args.device, non_blocking = True), y.to(args.device, non_blocking = True), ylen.to(args.device, non_blocking = True)
			try:
				#TODO check nan values in tensors, they can break running_stats in bn
				log_probs, olen, loss = map(model(x, xlen, y = y, ylen = ylen).get, ['log_probs', 'olen', 'loss'])
				oom_handler.reset()
			except:
				## TODO remove args.world_size > 1 after OOM recover fix for DDP
				if args.world_size > 1 or not oom_handler.try_recover(model.parameters(), _print=utils.get_root_logger_print(level=logging.ERROR)):
					raise
			example_weights = ylen[:, 0]
			loss, loss_cur = (loss * example_weights).mean() / args.train_batch_accumulate_iterations, loss.mean()
			entropy = models.entropy(log_probs[0], olen[0], dim=1).mean()

			if args.world_size > 1:
				dist.all_reduce(loss_cur, op=dist.ReduceOp.SUM)
				dist.all_reduce(entropy, op=dist.ReduceOp.SUM)
				loss_cur = loss_cur / args.world_size
				entropy = entropy / args.world_size

			perf.update(dict(entropy = entropy.item()))
			perf.update(dict(loss_BT_normalized = loss_cur.item()))

			toc_fwd = time.time()
			#TODO: inf/nan still corrupts BN stats
			if not (torch.isinf(loss_cur) or torch.isnan(loss_cur)):
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
					optimizer.zero_grad()
					scheduler.step(iteration)

				if iteration > 0 and iteration % args.log_iteration_interval == 0:
					if args.world_size > 1:
						perf.update(utils.compute_cuda_memory_stats(devices = [args.local_rank]), prefix = 'performance')
						tensorboard_sink.perf(perf.default(), iteration, rank = args.rank)
					else:
						perf.update(utils.compute_cuda_memory_stats(), prefix = 'performance')
						tensorboard_sink.perf(perf.default(), iteration, train_dataset_name)
					if args.rank == 0:
						tensorboard_sink.train_stats(perf.default(), iteration, train_dataset_name)
						tensorboard_sink.weight_stats(iteration, model, args.log_weight_distribution)
					logfile_sink.log(perf.default(), iteration, train_dataset_name)
			else:
				_print('Loss corruption detected, skip step.')
				if (torch.isinf(loss) or torch.isnan(loss)):
					logging.getLogger().error(f'Loss value is corrupted! {args.experiment_id} | epoch: {epoch:02d} iter: [{batch_idx: >6d} / {len(train_data_loader)} {iteration: >6d}] | Rank: {args.rank} | Loss: {loss.item()}')
			toc_bwd = time.time()

			time_ms_data, time_ms_fwd, time_ms_bwd, time_ms_model = map(lambda sec: sec * 1000, [toc_data - tic, toc_fwd - toc_data, toc_bwd - toc_fwd, toc_bwd - toc_data])
			perf.update(dict(time_ms_data = time_ms_data, time_ms_fwd = time_ms_fwd, time_ms_bwd = time_ms_bwd, time_ms_iteration = time_ms_data + time_ms_model), prefix = 'performance')
			perf.update(dict(input_B = x.shape[0], input_T = x.shape[-1]), prefix = 'performance')
			print_left = f'{args.experiment_id} | epoch: {epoch:02d} iter: [{batch_idx: >6d} / {len(train_data_loader)} {iteration: >6d}] {"x".join(map(str, x.shape))}'
			print_right = 'ent: <{avg_entropy:.2f}> loss: {cur_loss_BT_normalized:.2f} <{avg_loss_BT_normalized:.2f}> time: {performance_cur_time_ms_data:.2f}+{performance_cur_time_ms_fwd:4.0f}+{performance_cur_time_ms_bwd:4.0f} <{performance_avg_time_ms_iteration:.0f}> | lr: {cur_lr:.5f}'.format(**perf.default())
			_print(print_left, print_right)
			iteration += 1
			sampler.batch_idx += args.world_size

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
					tensorboard_sink,
					logfile_sink,
					epoch,
					iteration
				)

			if iteration and args.iterations and iteration >= args.iterations:
				return

			if args.iterations_per_epoch and iteration > 0 and iteration % args.iterations_per_epoch == 0:
				break

			tic = time.time()

		sampler.batch_idx = 0
		_print('Epoch time', (time.time() - time_epoch_start) / 60, 'minutes')
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
				tensorboard_sink,
				logfile_sink,
				epoch + 1,
				iteration
			)


def init_process(args):
	""" Initialize the distributed environment. """
	utils.set_random_seed(args.seed)
	if args.backend is None:
		args.backend = 'nccl' if args.device == 'cuda' else 'gloo'

	if args.device == 'cuda':
		torch.backends.cudnn.deterministic = True
		torch.cuda.set_device(args.local_rank)
		assert args.backend == 'nccl', 'CUDA supports only nccl backend'
		# synchronization timeout do not work without this env variable
		if args.synchronization_timeout:
			os.environ['NCCL_BLOCKING_WAIT'] = '1'
	else:
		assert args.backend != 'nccl', f'nccl backend do not support this device {args.device}'

	print(f"Initialize worker with rank {args.rank} in world size {args.world_size} at {args.master_ip}:{args.master_port}")
	dist.init_process_group(args.backend,
							init_method = f"tcp://{args.master_ip}:{args.master_port}",
							rank = args.rank,
							world_size = args.world_size,
							timeout = datetime.timedelta(seconds = args.synchronization_timeout) if args.synchronization_timeout else datetime.timedelta(minutes = 30))
	main(args)


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
	# DDP settings
	parser.add_argument('--start-rank', type = int, default = 0, help = 'processes ranks equal: start_rank+local_rank')
	parser.add_argument('--local-ranks', type = int, nargs = '+', default = [0], help ='processes local ranks, equals to cuda device id')
	parser.add_argument('--world-size', type = int, default = 1, help = 'amount of processes')
	parser.add_argument('--backend', choices = ['nccl', 'gloo'], help = 'processes communication backend')
	parser.add_argument('--master-ip', default = '0.0.0.0', help = 'DDP MASTER_ADDR')
	parser.add_argument('--master-port', default = '12345', help = 'DDP MASTER_PORT')
	parser.add_argument('--synchronize-bn', action = 'store_true', help = 'replace all instance of torch.nn.modules.batchnorm._BatchNorm to SyncBatchNorm')
	parser.add_argument('--synchronization-timeout', type = int, help = 'DDP synchronization method timeout in seconds')

	args = parser.parse_args()

	try:
		if args.world_size > 1:
			processes = []
			assert args.num_workers % args.world_size == 0, f'''--num-workers should be must be a multiple of --world-size,
																but now: --num-workers {args.num_workers} --world-size {args.world_size}, 
																num-workers % world-size should be 0, but get {args.num_workers % args.world_size}'''
			args.num_workers = int(args.num_workers / args.world_size)
			args.train_batch_size = int(args.train_batch_size / args.world_size)
			args.val_batch_size = int(args.val_batch_size / args.world_size)
			for rank in args.local_ranks:
				args.local_rank = rank
				args.rank = args.start_rank + rank
				p = torch.multiprocessing.Process(target=init_process, args=(args,))
				p.start()
				processes.append(p)

			for p in processes:
				p.join()
		else:
			args.local_rank = 0
			args.rank = 0
			main(args)
	except Exception as e:
		logging.getLogger().critical(e, exc_info=True)
		raise
