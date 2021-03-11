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
import datetime
import subprocess
import time
import multiprocessing
import torch.utils.data
import torch.utils.tensorboard
import onnx
import onnx.tools.net_drawer
import apex
import datasets
import transcript_generators
import metrics
import models
import optimizers
import torch
import vis
import utils
import transcripts
import perf
import itertools
import text_tokenizers
import text_processing
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
	

def apply_model(data_loader, model, generator, text_pipelines, device, oom_handler, forward_x_only=False):
	for meta, s, x, xlen, y, ylen in data_loader:
		x, xlen, y, ylen = x.to(device, non_blocking = True), xlen.to(device, non_blocking = True), y.to(device, non_blocking = True), ylen.to(device, non_blocking = True)
		with torch.no_grad():
			try:
				infer_kwargs = dict(x = x) if forward_x_only else dict(x = x, xlen = xlen, y = y, ylen = ylen)
				logits, log_probs, olen, loss = map(model(**infer_kwargs).get, ['logits', 'log_probs', 'olen', 'loss'])

				if loss is None:
					loss = torch.tensor([float.inf] * x.shape[0])

				oom_handler.reset()
			except:
				## TODO remove args.world_size > 1 after OOM recover fix for DDP
				if args.world_size > 1 or not oom_handler.try_recover(model.parameters(), _print = utils.get_root_logger_print(level = logging.ERROR)):
					raise

		entropy_char, *entropy_bpe = list(map(models.entropy, log_probs, olen))
		# TODO: use tokenizer.eps_id as the value for models.weighted_mean_entropy eps_id argument
		uncertainty_char, *uncertainty_bpe = list(map(models.weighted_mean_entropy, log_probs, olen))
		hyp = []
		for pipeline, lp, o in zip(text_pipelines, log_probs, olen):
			generated_transcripts = generator.generate(tokenizer = pipeline.tokenizer,
			                                           log_probs = lp,
			                                           begin = torch.zeros([lp.shape[0]], dtype = torch.float, device = 'cpu'),
			                                           end = torch.zeros([lp.shape[0]], dtype = torch.float, device = 'cpu'),
			                                           output_lengths = o,
			                                           time_stamps = None,
			                                           segment_text_key = 'hyp')
			hyp.append([transcripts.join(hyp = alternatives[0]) for alternatives in generated_transcripts])

		logits = list(map(models.unpad, logits, olen))
		y = list(map(models.unpad, y, ylen))
		yield meta, loss.cpu(), entropy_char.cpu(), uncertainty_char.cpu(), hyp, logits, y


def evaluate_model(
	args,
	val_data_loaders,
	model,
	generator,
	text_pipelines,
	error_analyzer,
	optimizer = None,
	sampler = None,
	tensorboard_sink = None,
	logfile_sink = None,
	epoch = None,
	iteration = None,
	forward_x_only=False,
):
	_print = utils.get_root_logger_print()

	training = epoch is not None and iteration is not None
	oom_handler = utils.OomHandler(max_retries=args.oom_retries)
	for val_dataset_name, val_data_loader in val_data_loaders.items():
		if args.world_size > 1:
			# synchronize process, because rank 0 process do result analisys and io
			dist.barrier()
		_print(f'\n{val_dataset_name}@{iteration}: batches: {len(val_data_loader)}, examples: {len(val_data_loader.dataset)}')
		analyze = args.analyze == [] or (args.analyze is not None and val_dataset_name in args.analyze)

		model.eval()
		if args.adapt_bn:
			models.reset_bn_running_stats_(model)
			for _ in apply_model(val_data_loader, model, generator, text_pipelines, args.device, oom_handler):
				pass
		model.eval()

		tic = time.time()
		ref_, hyp_, audio_path_, loss_, entropy_, uncertainty_ = [], [], [], [], [], []
		for batch_idx, (meta, loss, entropy, uncertainty, hyp, logits, y) \
			in enumerate(apply_model(val_data_loader, model, generator, text_pipelines, args.device, oom_handler, forward_x_only=forward_x_only)):
			loss_.extend(loss.tolist())
			entropy_.extend(entropy.tolist())
			uncertainty_.extend(uncertainty.tolist())
			audio_path_.extend(m['audio_path'] for m in meta)
			ref_.extend([pipeline.preprocess(m['ref']) for pipeline in text_pipelines] for m in meta)
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
				return string_storage.tolist()
			_print(f"Synchronize validation results started")
			sync_tic = time.time()
			loss_ = sync_float_list(loss_)
			_print(f'loss sync time {time.time() - sync_tic:.1f} sec');sync_tic = time.time()
			entropy_ = sync_float_list(entropy_)
			_print(f'entropy sync time {time.time() - sync_tic:.1f} sec');sync_tic = time.time()
			audio_path_ = sync_string_list(audio_path_)
			_print(f'audio_path sync time {time.time()  - sync_tic:.1f} sec');sync_tic = time.time()
			ref_by_pipelines = list(map(list, zip(*ref_)))
			for i in range(len(ref_by_pipelines)):
				ref_by_pipelines[i] = sync_string_list(ref_by_pipelines[i])
			ref_ = list(map(list, zip(*ref_by_pipelines)))
			_print(f'ref sync time {time.time() - sync_tic:.1f} sec');sync_tic = time.time()
			hyps_by_pipelines = list(map(list, zip(*hyp_)))
			for i in range(len(hyps_by_pipelines)):
				hyps_by_pipelines[i] = sync_string_list(hyps_by_pipelines[i])
			hyp_ = list(map(list, zip(*hyps_by_pipelines)))
			_print(f'hyp sync time {time.time()- sync_tic:.1f} sec')
			time_sec_val_sync_results = time.time() - toc_apply_model
			perf.update(dict(time_sec_val_sync_results=time_sec_val_sync_results), prefix=f'datasets_{val_dataset_name}')
			_print(f"Synchronize validation results {time_sec_val_sync_results:.1f} sec")
			if args.rank != 0:
				continue

		#TODO add duration to extra info dict
		analyze_args_gen = (
			(
				hyp,
				ref,
				pipeline.postprocess,
				analyze,
				dict(
					labels_name=pipeline.name,
					audio_path=audio_path,
					audio_name=transcripts.audio_name(audio_path),
					loss=loss,
					entropy=entropy,
					uncertainty=uncertainty,
				),
			)
			for ref_tuple, hyp_tuple, audio_path, loss, entropy, uncertainty
			in zip(ref_, hyp_, audio_path_, loss_, entropy_, uncertainty_)
			for pipeline, ref, hyp in zip(text_pipelines, ref_tuple, hyp_tuple)
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
			_print(f'{val_dataset_name}@{iteration}: {i // len(text_pipelines)} / {len(audio_path_)} | {args.experiment_id}')
			# TODO: don't forget to fix aligned hyp & ref output!
			_print('{labels_name} AUDIO_NAME: {audio_name}'.format(**t))
			_print('{labels_name} REF: "{ref}"'.format(**t))
			_print('{labels_name} HYP: "{hyp}"'.format(**t))
			_print('{labels_name} WER: {wer:.02%} | CER: {cer:.02%} | UNCERTAINTY: {uncertainty:.6f}\n'.format(**t))

		transcripts_path = os.path.join(
			args.experiment_dir,
			args.train_transcripts_format
			.format(val_dataset_name = val_dataset_name, epoch = epoch, iteration = iteration)
		) if training else os.path.join(
			args.experiment_dir,
			args.val_transcripts_format
			.format(val_dataset_name = val_dataset_name, decoder = args.decoder)
		)
		for i, pipeline in enumerate(text_pipelines):
			transcript_by_label = transcript[i:: len(text_pipelines)]
			aggregated = error_analyzer.aggregate(transcript_by_label, sep = '__', defaults = dict(words_easy_errors_easy__cer_pseudo = -1, mer_wordwise = -1, hyp_vocabness = -1, ref_vocabness = -1))
			if analyze:
				with open(f'{transcripts_path}.errors.csv', 'w') as f:
					f.writelines('{hyp},{ref},{error_tag}\n'.format(**w) for w in aggregated['errors']['words'])

			_print('errors', aggregated['errors']['distribution'])
			_print('cer', metrics.quantiles(t['cer'] for t in transcript_by_label))
			_print('loss', metrics.quantiles(t['loss'] for t in transcript_by_label))
			_print(
				f'{args.experiment_id} {val_dataset_name} {pipeline.name} |',
				f'epoch {epoch} iter {iteration} |' if training else '',
				f'{transcripts_path}.json |' if args.output_json else '',
				f'{transcripts_path}.csv |' if args.output_csv else '',
				('Entropy: {entropy:.02f} Loss: {loss:.02f} | WER:  {wer:.02%} CER: {cer:.02%} [{words_easy_errors_easy__cer_pseudo:.02%}],  MER: {mer_wordwise:.02%} V: {hyp_vocabness:.02%}/{ref_vocabness:.02%}\n')
				.format(**aggregated)
			)
			
			if training:
				perf.update(dict(
						wer = aggregated['wer'],
						cer = aggregated['cer'],
						loss = aggregated['loss']
					), prefix = f'datasets_val_{val_dataset_name}_{pipeline.name}'
				)
				tensorboard_sink.val_stats(iteration, val_dataset_name, pipeline.name, perf.default())

		if args.output_json:
			with open(transcripts_path + '.json', 'w') as f:
				json.dump(transcript, f, ensure_ascii = False, indent = 2, sort_keys = True)
			if analyze:
				vis.errors([transcripts_path + '.json'], debug_audio = args.vis_errors_audio)

		if args.output_csv:
			with open(transcripts_path + '.csv', 'w') as f:
				f.write(args.csv_sep.join(args.csv_columns) + '\n')
				f.writelines(args.csv_sep.join(str(t[column]) for column in args.csv_columns) + '\n' for t in transcript)

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
				generators = [pipeline.name for pipeline in text_pipelines]
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
		experiment_name = args.experiment_name
	).replace('e-0', 'e-').rstrip('_')
	if checkpoint and 'experiment_id' in checkpoint['args'] and not args.experiment_name:
		args.experiment_id = checkpoint['args']['experiment_id']
	args.experiment_dir = args.experiment_dir.format(
		experiments_dir = args.experiments_dir, experiment_id = args.experiment_id
	)

	os.makedirs(args.experiment_dir, exist_ok = True)

	if args.log_json:
		args.log_json = os.path.join(args.experiment_dir, f'log.node{args.rank}.json')

	if args.rank == 0:
		log_level = logging.INFO
	else:
		log_level = logging.ERROR

	log_path = os.path.join(args.experiment_dir, 'log.txt')
	log_mode = 'w'

	if checkpoint:
		args.model, args.num_input_features, args.sample_rate, args.window, args.window_size, args.window_stride = map(checkpoint['args'].get, ['model', 'num_input_features', 'sample_rate', 'window', 'window_size', 'window_stride'])
		if len(args.train_data_path) > 0:
			log_mode = 'a'
	utils.set_up_root_logger(log_path, mode = log_mode, level = log_level)
	logfile_sink = JsonlistSink(args.log_json, mode = log_mode)


	_print = utils.get_root_logger_print()
	_print('\n', 'Arguments:', args)
	_print(f'"CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", default = "")}"')
	_print(f'"CUDA_LAUNCH_BLOCKING={os.environ.get("CUDA_LAUNCH_BLOCKING", default = "")}"')
	_print('Experiment id:', args.experiment_id, '\n')
	if args.dry:
		return
	if args.cudnn == 'benchmark':
		torch.backends.cudnn.benchmark = True

	text_config = json.load(open(args.text_config))
	lang = text_config['lang']
	text_pipelines = []
	for pipeline_name in args.text_pipelines:
		text_pipelines.append(text_processing.ProcessingPipeline.make(text_config, pipeline_name))

	validation_postprocessors = {name: text_processing.TextPostprocessor(**config) for name, config in text_config['postprocess'].items()}

	frontend = getattr(models, args.frontend)(
		out_channels = args.num_input_features,
		sample_rate = args.sample_rate,
		window_size = args.window_size,
		window_stride = args.window_stride,
		window = args.window,
		dither = args.dither,
		dither0 = args.dither0,
		stft_mode = 'conv' if args.onnx and not args.onnx_validate else None,
		extra_args = frontend_extra_args
	)
	model = getattr(models, args.model)(
		num_input_features = args.num_input_features,
		num_classes = [pipeline.tokenizer.vocab_size for pipeline in text_pipelines],
		dropout = args.dropout,
		decoder_type = 'bpe' if any(isinstance(pipeline.tokenizer, text_tokenizers.BPETokenizer) for pipeline in text_pipelines) else None,
		frontend = frontend if args.onnx or args.frontend_in_model else None,
		use_jited_function = args.onnx is None,
		**(dict(inplace = False, dict = lambda logits, log_probs, olen, **kwargs: logits[0]) if args.onnx and not args.onnx_validate else {})
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

	if args.onnx and not args.onnx_validate:
		torch.set_grad_enabled(False)
		model.eval()
		model.to(args.device)
		model.fuse_conv_bn_eval()

		if args.fp16:
			model = models.InputOutputTypeCast(model.to(torch.float16), dtype = torch.float16)

		xlen = None
		if args.onnx_waveform_input:
			waveform_input = torch.load(args.onnx_waveform_input, map_location=args.device).squeeze(1)
			# NOTE about squeeze(1). If the waveform was exported with tree dimensions shape [B,1,T] and center shape does not contains any data.
		else:
			xlen = torch.rand(args.onnx_sample_batch_size, device = args.device)
			waveform_input = torch.rand(args.onnx_sample_batch_size, args.onnx_sample_time, device = args.device)

		torch_logits = model(waveform_input, xlen)

		torch.onnx.export(
			model, (waveform_input, xlen, ),
			args.onnx,
			verbose = False,
			opset_version = args.onnx_opset,
			export_params = args.onnx_export_params,
			do_constant_folding = True,
			input_names = ['x', 'xlen'],
			output_names = ['logits'],
			dynamic_axes = dict(x = {
				0: 'B', 1: 'T'
			}, logits = {
				0: 'B', 2: 't'
			}, xlen = {
				0: 'B'
			})
		)

		wrapper = models.OnnxWrapper(args.onnx, severity = 0 if args.verbose else None)
		wrapper_logits = wrapper(waveform_input, xlen)['logits'][0]

		print('pytorch with wrapper max difference: ', (torch_logits - wrapper_logits).abs().max())

		assert torch.allclose(
				torch_logits.cpu(),
				wrapper_logits.cpu(),
				**{'rtol': 1e-01, 'atol': 1e-02} if args.fp16 else {'rtol': 1e-02, 'atol': 1e-03}
		)

		if args.onnx_dot_file:
			model_def = onnx.load(args.onnx)

			pydot_graph = onnx.tools.net_drawer.GetPydotGraph(
					model_def.graph,
					name=model_def.graph.name,
					rankdir="TB",
					node_producer=onnx.tools.net_drawer.GetOpNodeProducer(
							"docstring",
							color="yellow",
							fillcolor="yellow",
							style="filled"))
			pydot_graph.write_dot(args.onnx_dot_file)
			subprocess.run(['dot', '-O', '-Gdpi=300', '-Tpng', args.onnx_dot_file])
		return

	perf.init_default(loss=dict(K=50, max=1000), memory_cuda_allocated=dict(K=50), entropy=dict(K=4), time_ms_iteration=dict(K=50, max=10_000), lr=dict(K=50, max=1))

	val_config = json.load(open(args.val_config)) if os.path.exists(args.val_config) else {}
	word_tags = json.load(open(args.word_tags)) if os.path.exists(args.word_tags) else {}
	for word_tag, words in val_config.get('word_tags', {}).items():
		word_tags[word_tag] = word_tags.get(word_tag, []) + words
	vocab = set(map(str.strip, open(args.vocab))) if os.path.exists(args.vocab) else set()
	stemmer = text_processing.Stemmer(lang)
	error_analyzer = metrics.ErrorAnalyzer(metrics.WordTagger(stemmer = stemmer, vocab = vocab, word_tags = word_tags), metrics.ErrorTagger(), val_config.get('error_analyzer', {}), validation_postprocessors)

	val_frontend = frontend

	val_data_loaders = {}
	for val_data_path in args.val_data_path:
		val_dataset = datasets.AudioTextDataset(val_data_path, text_pipelines, args.sample_rate,
												frontend = val_frontend if not args.frontend_in_model else None,
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

		val_data_loaders[os.path.basename(os.path.normpath(val_data_path))] = val_data_loader

	generator = transcript_generators.GreedyCTCGenerator()
	model.to(args.device)

	if not args.train_data_path:
		model.eval()
		if not args.adapt_bn:
			model.fuse_conv_bn_eval()
		if args.world_size > 1:
			model, *_ = models.distributed_data_parallel_and_autocast(model, args.local_rank, opt_level = args.fp16, synchronize_bn = args.synchronize_bn, keep_batchnorm_fp32 = args.fp16_keep_batchnorm_fp32)
		elif args.device != 'cpu':
			model, *_ = models.data_parallel_and_autocast(model, opt_level = args.fp16, keep_batchnorm_fp32 = args.fp16_keep_batchnorm_fp32)

		model = models.OnnxWrapper(args.onnx) if args.onnx else model

		evaluate_model(args, val_data_loaders, model, generator, text_pipelines, error_analyzer, forward_x_only=args.forward_x_only)
		return

	model.freeze(backbone = args.freeze_backbone, decoder0 = args.freeze_decoder, frontend = args.freeze_frontend)

	train_frontend = frontend
	tic = time.time()
	if args.local_rank == 0:
		train_dataset = datasets.AudioTextDataset(
			args.train_data_path,
			text_pipelines,
			args.sample_rate,
			frontend = train_frontend if not args.frontend_in_model else None,
			min_duration = args.min_duration,
			max_duration = args.max_duration,
			time_padding_multiple = args.batch_time_padding_multiple,
			bucket_fn = lambda example: int(
				math.ceil(
					((example[-1]['end'] - example[0]['begin']) / args.window_stride + 1) / args.batch_time_padding_multiple
				)
			),
			pop_meta = True,
			_print = _print
		)
		if args.world_size > 1:
			cache_path = f'{args.experiment_dir}/dataset_cache.pt'
			if os.path.exists(cache_path):
				os.remove(cache_path)
			torch.save(train_dataset.state_dict(), cache_path)

	if args.world_size > 1:
		# waiting for dataset cache serialization
		dist.barrier()

	if args.local_rank != 0:
		train_dataset = datasets.AudioTextDataset(
			[],
			text_pipelines,
			args.sample_rate,
			frontend=train_frontend if not args.frontend_in_model else None,
			min_duration=args.min_duration,
			max_duration=args.max_duration,
			time_padding_multiple=args.batch_time_padding_multiple,
			bucket_fn =lambda example: int(
				math.ceil(
					((example[-1]['end'] - example[0]['begin']) / args.window_stride + 1) / args.batch_time_padding_multiple
				)
			),
			pop_meta=True,
			_print=_print
		)
		train_dataset.load_state_dict(torch.load(f'{args.experiment_dir}/dataset_cache.pt'))

	if args.world_size > 1:
		# waiting for dataset cache loading
		dist.barrier()

	_print('Time train dataset created:', time.time() - tic, 'sec')
	train_dataset_name = '_'.join(map(os.path.basename, args.train_data_path))
	tic = time.time()
	sampler = datasets.BucketingBatchSampler(train_dataset, batch_size = args.train_batch_size, world_size = args.world_size)
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
					generator,
					text_pipelines,
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
				generator,
				text_pipelines,
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
		default = 'transcripts_{val_dataset_name}_{decoder}',
		help = 'save transcripts at validation'
	)
	parser.add_argument(
		'--train-transcripts-format',
		default = 'transcripts_{val_dataset_name}_epoch{epoch:02d}_iter{iteration:07d}'
	)
	parser.add_argument('--output-json', type = utils.str2bool, nargs = '?', const = True, default = True)
	parser.add_argument('--output-csv', action='store_true')
	parser.add_argument('--csv-sep', default=',')
	parser.add_argument(
		'--csv-columns', nargs='+', default=['labels_name', 'audio_path', 'audio_name', 'ref', 'hyp', 'ref_orig', 'hyp_orig', 'cer', 'wer', 'loss', 'entropy']
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
		'{model}_{optimizer}_lr{lr:.0e}_wd{weight_decay:.0e}_bs{train_batch_size}_{experiment_name}'
	)
	parser.add_argument('--experiment-name', '--name', default = '')
	parser.add_argument('--comment', default = '')
	parser.add_argument('--dry', action = 'store_true')
	parser.add_argument('--train-batch-accumulate-iterations', type = int, default = 1, help = 'number of gradient accumulation steps')
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
	parser.add_argument('--onnx-validate', action = 'store_true', help = 'flag is need to set if you validate onnx model with OnnxWrapper without export from pytorch',)
	parser.add_argument('--forward-x-only', action = 'store_true', help = 'flag for validation mode with inference without xlen model(x) instead of model(x, xlen ...), needed to debug masking',)
	parser.add_argument('--onnx-waveform-input', type=str)
	parser.add_argument('--onnx-dot-file', type=str)
	parser.add_argument('--onnx-sample-batch-size', type = int, default = 16)
	parser.add_argument('--onnx-sample-time', type = int, default = 1024)
	parser.add_argument('--onnx-opset', type = int, default = 12, choices = [9, 10, 11, 12, 13])
	parser.add_argument('--onnx-export-params', type = bool, default = True)
	parser.add_argument('--dropout', type = float, default = 0.2)
	parser.add_argument('--githttp')
	parser.add_argument('--vis-errors-audio', action = 'store_true')
	parser.add_argument('--adapt-bn', action = 'store_true')
	parser.add_argument('--vocab', default = 'data/vocab_word_list.txt')
	parser.add_argument('--word-tags', default = 'data/word_tags.json')
	parser.add_argument('--text-config', default = 'configs/ru_text_config.json')
	parser.add_argument('--text-pipelines', nargs = '+', help = 'text processing pipelines (names should be defined in text-config)', default = ['char_legacy'])
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
