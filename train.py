import os
import sys
import json
import math
import time
import shutil
import random
import itertools
import argparse
import importlib
import torch
import torch.utils.data
import torch.utils.tensorboard
import torch.nn as nn
import torch.nn.functional as F
import apex
import onnxruntime
import dataset
import transforms
import decoders
import metrics
import models
import optimizers
import vis

def set_random_seed(seed):
	for set_random_seed in [random.seed, torch.manual_seed] + ([torch.cuda.manual_seed_all] if torch.cuda.is_available() else []):
		set_random_seed(seed)

def traineval(args):
	checkpoints = [torch.load(checkpoint_path, map_location = 'cpu') for checkpoint_path in args.checkpoint]
	checkpoint = (checkpoints + [{}])[0]
	if len(checkpoints) > 1:
		checkpoint['model_state_dict'] = {k : sum(c['model_state_dict'][k] for c in checkpoints) / len(checkpoints) for k in checkpoint['model_state_dict']}

	args.experiment_id = args.experiment_id.format(model = args.model, train_batch_size = args.train_batch_size, optimizer = args.optimizer, lr = args.lr, weight_decay = args.weight_decay, time = time.strftime('%Y-%m-%d_%H-%M-%S'), experiment_name = args.experiment_name, bpe = 'bpe' if args.bpe else '', train_waveform_transform = f'aug{args.train_waveform_transform[0]}{args.train_waveform_transform_prob or ""}' if args.train_waveform_transform else '', train_feature_transform = f'aug{args.train_feature_transform[0]}{args.train_feature_transform_prob or ""}' if args.train_feature_transform else '').replace('e-0', 'e-').rstrip('_')
	if 'experiment_id' in checkpoint and not args.experiment_name:
		args.experiment_id = checkpoint['experiment_id']

	args.experiment_dir = args.experiment_dir.format(experiments_dir = args.experiments_dir, experiment_id = args.experiment_id)
	args.lang, args.model, args.num_input_features = checkpoint.get('lang', args.lang), checkpoint.get('model', args.model), checkpoint.get('num_input_features', args.num_input_features)
	if not args.train_data_path and checkpoint and 'args' in checkpoint:
		args.sample_rate, args.window, args.window_size, args.window_stride = map(checkpoint['args'].get, ['sample_rate', 'window', 'window_size', 'window_stride'])

	print('\n', 'Arguments:', args)
	print('\n', 'Experiment id:', args.experiment_id, '\n')
	if args.dry:
		return
	set_random_seed(args.seed)
	if args.cudnn == 'benchmark':
		torch.backends.cudnn.benchmark = True

	lang = importlib.import_module(args.lang)
	labels = [dataset.Labels(lang, name = 'char')] + [dataset.Labels(lang, bpe = bpe, name = f'bpe{i}') for i, bpe in enumerate(args.bpe)]
	model = getattr(models, args.model)(num_input_features = args.num_input_features, num_classes = list(map(len, labels)), dropout = args.dropout, **(dict(inplace = False, dict = lambda logits, output_lengths: (logits, output_lengths)) if args.onnx else {}))

	if args.onnx:
		torch.set_grad_enabled(False)
		model.to(args.device)
		if args.fp16:
			model = apex.amp.initialize(model, opt_level = args.fp16, keep_batchnorm_fp32 = args.fp16_keep_batchnorm_fp32)
		model.eval()
		model.fuse_conv_bn_eval()
		#model = torch.nn.DataParallel(model)
		sample_batch_size, sample_time = 16, 128
		input_, input_lengths_fraction_ = torch.rand(sample_batch_size, args.num_input_features, sample_time, device = args.device), torch.ones(sample_batch_size, device = args.device)
		logits, output_lengths = model(input_, input_lengths_fraction_)

		torch.onnx.export(model, (input_, input_lengths_fraction_), args.onnx, opset_version = 11, do_constant_folding = True, input_names = ['x', 'xlen'], output_names = ['logits', 'output_lengths'], dynamic_axes = dict(x = {0 : 'B'}, xlen = {0 : 'B'}, logits = {0 : 'B'}, output_lengths = {0 : 'B'}))

		ort_session = onnxruntime.InferenceSession(args.onnx)
		ort_inputs = dict(x = input_.cpu().numpy(), xlen = input_lengths_fraction_.cpu().numpy())
		logits_, output_lengths_ = ort_session.run(None, ort_inputs)
		assert torch.allclose(logits[0].cpu(), torch.from_numpy(logits_), rtol=1e-03, atol=1e-05)
		assert torch.allclose(output_lengths[0].cpu(), torch.from_numpy(output_lengths_))
		return

	make_transform = lambda name_args, prob: None if not name_args else getattr(transforms, name_args[0])(*name_args[1:]) if prob is None else getattr(transforms, name_args[0])(prob, *name_args[1:]) if prob > 0 else None
	val_waveform_transform = make_transform(args.val_waveform_transform, args.val_waveform_transform_prob)
	val_feature_transform = make_transform(args.val_feature_transform, args.val_feature_transform_prob)
	if args.val_waveform_transform_debug_dir:
		args.val_waveform_transform_debug_dir = os.path.join(args.val_waveform_transform_debug_dir, str(val_waveform_transform) if isinstance(val_waveform_transform, transforms.RandomCompose) else val_waveform_transform.__class__.__name__)
		os.makedirs(args.val_waveform_transform_debug_dir, exist_ok = True)
	
	val_data_loaders = {os.path.basename(val_data_path) : torch.utils.data.DataLoader(dataset.AudioTextDataset(val_data_path, sample_rate = args.sample_rate, window_size = args.window_size, window_stride = args.window_stride, window = args.window, labels = labels, num_input_features = args.num_input_features, waveform_transform = val_waveform_transform, waveform_transform_debug_dir = args.val_waveform_transform_debug_dir, feature_transform = val_feature_transform), num_workers = args.num_workers, collate_fn = dataset.collate_fn, pin_memory = True, shuffle = False, batch_size = args.val_batch_size, worker_init_fn = set_random_seed, timeout = args.timeout) for val_data_path in args.val_data_path}
	decoder = [decoders.GreedyDecoder() if args.decoder == 'GreedyDecoder' else decoders.BeamSearchDecoder(labels[0], lm_path = args.lm, beam_width = args.beam_width, beam_alpha = args.beam_alpha, beam_beta = args.beam_beta, num_workers = args.num_workers, topk = args.decoder_topk)] + [decoders.GreedyDecoder() for bpe in args.bpe]

	def apply_model(data_loader):
		for dataset_name_, audio_path_, reference_, input_, input_lengths_fraction_, targets_, target_length_ in data_loader:
			with torch.no_grad():
				log_probs, output_lengths, loss = map(model(input_.to(args.device), input_lengths_fraction_.to(args.device), y = targets_.to(args.device), ylen = target_length_.to(args.device)).get, ['log_probs', 'output_lengths', 'loss'])
			entropy_char, *entropy_bpe = list(map(models.entropy, log_probs, output_lengths))
			decoded = [list(map(l.decode, d.decode(lp, o))) for l, d, lp, o in zip(labels, decoder, log_probs, output_lengths)]
			log_probs = list(map(models.unpad, log_probs, output_lengths))
			yield audio_path_, reference_, loss.cpu(), entropy_char.cpu(), decoded, log_probs

	def evaluate_transcript(align, labels, transcript, reference, loss, entropy, audio_path):
		transcript, reference = min((metrics.cer(t, r), (t, r)) for r in reference.split(';') for t in (transcript if isinstance(transcript, list) else [transcript]))[1]
		transcript, reference = map(labels.postprocess_transcript, [transcript, reference])
		cer, wer = metrics.cer(transcript, reference), metrics.wer(transcript, reference) 
		transcript_phonetic, reference_phonetic = [labels.postprocess_transcript(s, phonetic_replace_groups = lang.PHONETIC_REPLACE_GROUPS) for s in [transcript, reference]]
		per = metrics.cer(transcript_phonetic, reference_phonetic)
		aligned = dict(zip(['_hyp_aligned', '_ref_aligned'], metrics.align(transcript, reference))) if align else {}
		return dict(ref_phonetic = reference_phonetic, hyp_phonetic = transcript_phonetic, ref = reference, hyp = transcript, cer = cer, wer = wer, per = per, labels = labels.name, audio_file_name = os.path.basename(audio_path), audio_path = audio_path, entropy = entropy, loss = loss, **aligned)

	def evaluate_model(val_data_loaders, epoch = None, iteration = None):
		training = epoch is not None and iteration is not None
		model.eval()
		columns = {}
		for val_dataset_name, val_data_loader in val_data_loaders.items():
			print(f'\n{val_dataset_name}@{iteration}')
			ref_tra_, logits_ = [], []
			align = args.align is None or val_dataset_name in args.align
			for batch_idx, (audio_paths, references, loss, entropy, decoded, logits) in enumerate(apply_model(val_data_loader)):
				logits_.append(logits if not training and args.logits else None)
				stats = [[evaluate_transcript(align, l, t, *zipped[:4]) for l, t in zip(labels, zipped[4:])] for zipped in zip(references, loss.tolist(), entropy.tolist(), audio_paths, *decoded)]
				for r in itertools.chain(*stats) if args.verbose else []:
					print(f'{val_dataset_name}@{iteration}:', batch_idx , '/', len(val_data_loader), '|', args.experiment_id)
					print('REF: {labels} "{_ref_aligned}"'.format(**r, **(dict(_ref_aligned = r['ref']) if not align else {})))
					print('HYP: {labels} "{_hyp_aligned}"'.format(**r, **(dict(_hyp_aligned = r['hyp']) if not align else {})))
					print('WER: {labels} {wer:.02%} | CER: {cer:.02%}\n'.format(**r))
				ref_tra_.extend(stats)

			transcripts_path = os.path.join(args.experiment_dir, args.train_transcripts_format.format(val_dataset_name = val_dataset_name, epoch = epoch, iteration = iteration)) if training else args.val_transcripts_format.format(val_dataset_name = val_dataset_name, decoder = args.decoder)
			for r_ in zip(*ref_tra_):
				cer_avg, wer_avg, loss_avg, entropy_avg = [float(torch.tensor([r[k] for r in r_ if not math.isinf(r[k]) and not math.isnan(r[k])]).mean()) for k in ['cer', 'wer', 'loss', 'entropy']]
				labels_name = r_[0]['labels']
				print(f'{args.experiment_id} {val_dataset_name} {labels_name}', f'| epoch {epoch} iter {iteration}' if training else '', f'| {transcripts_path} | Entropy: {entropy_avg:.02f} Loss: {loss_avg:.02f} | WER:  {wer_avg:.02%} CER: {cer_avg:.02%}\n')
				
				columns[val_dataset_name + '_' + labels_name] = dict(cer_avg = cer_avg, wer_avg = wer_avg, loss_avg = loss_avg, entropy_avg = entropy_avg)
				if training:
					tensorboard.add_scalars('datasets/' + val_dataset_name + '_' + labels_name, dict(wer_avg = wer_avg * 100.0, cer_avg = cer_avg * 100.0, loss_avg = loss_avg), iteration) 
					tensorboard.flush()

			with open(transcripts_path, 'w') as f:
				json.dump([r for t in sorted(ref_tra_, key = lambda t: t[0]['cer'], reverse = True) for r in t], f, ensure_ascii = False, indent = 2, sort_keys = True)
			
			#if args.logits:
			#	logits_file_path = args.logits.format(val_dataset_name = val_dataset_name)
			#	print('Logits:', logits_file_path)
			#	torch.save(list(sorted([(r.update(dict(log_probs = l)) or r) for r, l in zip(ref_tra_, logits_)], key = lambda r: r['cer'], reverse = True)), logits_file_path)
		
		if args.exphtml:
			vis.expjson(args.exphtml, args.experiment_id, epoch = epoch, iteration = iteration, meta = vars(args), columns = columns, git_http = args.githttp)
			vis.exphtml(args.exphtml)
		
		if training and not args.checkpoint_skip:
			amp_state_dict = None # amp.state_dict()
			optimizer_state_dict = None # optimizer.state_dict()
			torch.save(dict(model = model.module.__class__.__name__, model_state_dict = model.module.state_dict(), optimizer_state_dict = optimizer_state_dict, amp_state_dict = amp_state_dict, scheduler_state_dict = scheduler.state_dict(), sampler_state_dict = sampler.state_dict(), epoch = epoch, iteration = iteration, args = vars(args), experiment_id = args.experiment_id, lang = args.lang, num_input_features = args.num_input_features, time = time.time()), os.path.join(args.experiment_dir, args.checkpoint_format.format(epoch = epoch, iteration = iteration)))

		model.train()
		torch.cuda.empty_cache()

	print(' Model capacity:', int(models.compute_capacity(model) / 1e6), 'million parameters\n')

	if checkpoint:
		model.load_state_dict(checkpoint['model_state_dict'], strict = False)
	
	model.to(args.device)

	if not args.train_data_path:
		if args.fp16:
			model = apex.amp.initialize(model, opt_level = args.fp16, keep_batchnorm_fp32 = args.fp16_keep_batchnorm_fp32)
		model.eval()
		model.fuse_conv_bn_eval()
		model = torch.nn.DataParallel(model)
		evaluate_model(val_data_loaders)
		return

	if args.freeze:
		model.freeze(backbone = True, decoder0 = True)

	train_waveform_transform = make_transform(args.train_waveform_transform, args.train_waveform_transform_prob)
	train_feature_transform = make_transform(args.train_feature_transform, args.train_feature_transform_prob)
	train_dataset = dataset.AudioTextDataset(args.train_data_path, sample_rate = args.sample_rate, window_size = args.window_size, window_stride = args.window_stride, window = args.window, labels = labels, num_input_features = args.num_input_features, waveform_transform = train_waveform_transform, feature_transform = train_feature_transform, max_duration = args.max_duration)
	train_dataset_name = '_'.join(map(os.path.basename, args.train_data_path))
	sampler = dataset.BucketingSampler(train_dataset, batch_size = args.train_batch_size, mixing = args.train_data_mixing)
	train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers = args.num_workers, collate_fn = dataset.collate_fn, pin_memory = True, batch_sampler = sampler, worker_init_fn = set_random_seed, timeout = args.timeout)
	optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay, nesterov = args.nesterov) if args.optimizer == 'SGD' else torch.optim.AdamW(model.parameters(), lr = args.lr, betas = args.betas, weight_decay = args.weight_decay) if args.optimizer == 'AdamW' else optimizers.NovoGrad(model.parameters(), lr = args.lr, betas = args.betas, weight_decay = args.weight_decay) if args.optimizer == 'NovoGrad' else apex.optimizers.FusedNovoGrad(model.parameters(), lr = args.lr, betas = args.betas, weight_decay = args.weight_decay) if args.optimizer == 'FusedNovoGrad' else None
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma = args.decay_gamma, milestones = args.decay_milestones) if args.scheduler == 'MultiStepLR' else optimizers.PolynomialDecayLR(optimizer, power = args.decay_power, decay_steps = len(train_data_loader) * args.decay_epochs, end_lr = args.decay_lr) if args.scheduler == 'PolynomialDecayLR' else torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.decay_step_size, gamma = args.decay_gamma) if args.scheduler == 'StepLR' else optimizers.NoopLR(optimizer) 
	iteration = 0
	if checkpoint:
		#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		iteration = 1 + checkpoint['iteration']
		if args.train_data_path == checkpoint['args']['train_data_path']:
			sampler.load_state_dict(checkpoint['sampler_state_dict'])
			scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
			if sampler.batch_idx + 1 == len(sampler):
				sampler.shuffle(epoch = sampler.epoch + 1, batch_idx = 0)
			else:
				#TODO: remove next line
				sampler.epoch = checkpoint['epoch']
				sampler.shuffle(epoch = sampler.epoch, batch_idx = sampler.batch_idx + 1)
				#sampler.batch_idx += 1
		else:
			sampler.shuffle(epoch = sampler.epoch + 1)

	if args.fp16:
		model, optimizer = apex.amp.initialize(model, optimizer, opt_level = args.fp16, keep_batchnorm_fp32 = args.fp16_keep_batchnorm_fp32)
	model = torch.nn.DataParallel(model)
	model.train()

	os.makedirs(args.experiment_dir, exist_ok = True)
	tensorboard_dir = os.path.join(args.experiment_dir, 'tensorboard')
	if checkpoint and args.experiment_name:
		tensorboard_dir_checkpoint = os.path.join(os.path.dirname(args.checkpoint[0]), 'tensorboard')
		if os.path.exists(tensorboard_dir_checkpoint) and not os.path.exists(tensorboard_dir):
			shutil.copytree(tensorboard_dir_checkpoint, tensorboard_dir)
	tensorboard = torch.utils.tensorboard.SummaryWriter(tensorboard_dir)
	with open(os.path.join(args.experiment_dir, args.args), 'w') as f:
		json.dump(vars(args), f, sort_keys = True, ensure_ascii = False, indent = 2)

	tic, toc_fwd, toc_bwd = time.time(), time.time(), time.time()
	loss_avg, entropy_avg, time_ms_avg, lr_avg = 0.0, 0.0, 0.0, 0.0
	moving_avg = lambda avg, x, max = 0, K = 50: (1. / K) * min(x, max) + (1 - 1. / K) * avg
	for epoch in range(sampler.epoch, args.epochs):
		time_epoch_start = time.time()
		for dataset_name_, audio_path_, reference_, input_, input_lengths_fraction_, targets_, target_length_ in train_data_loader:
			toc_data = time.time()
			lr = scheduler.get_lr()[0]; lr_avg = moving_avg(lr_avg, lr, max = 1)
			
			input_, input_lengths_, targets_, target_length_ = input_.to(args.device, non_blocking = True), input_lengths_fraction_.to(args.device, non_blocking = True), targets_.to(args.device, non_blocking = True), target_length_.to(args.device, non_blocking = True)
			log_probs, output_lengths, loss = map(model(input_, input_lengths_fraction_, y = targets_, ylen = target_length_).get, ['log_probs', 'output_lengths', 'loss'])
			example_weights = target_length_[:, 0]
			loss, loss_cur = (loss * example_weights).mean() / args.train_batch_accumulate_iterations, float(loss.mean())
			entropy = float(models.entropy(log_probs[0], output_lengths[0], dim = 1).mean()) 
			toc_fwd = time.time()
			if not (torch.isinf(loss) or torch.isnan(loss)):
				if args.fp16:
					with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
						scaled_loss.backward()
				else:
					loss.backward()

				if iteration % args.train_batch_accumulate_iterations == 0:
					torch.nn.utils.clip_grad_norm_(apex.amp.master_params(optimizer) if args.fp16 else model.parameters(), args.max_norm)
					optimizer.step()
					optimizer.zero_grad()
					scheduler.step()
				loss_avg = moving_avg(loss_avg, loss_cur, max = 1000)
				entropy_avg = moving_avg(entropy_avg, entropy, max = 4)
			toc_bwd = time.time()

			ms = lambda sec: sec * 1000
			time_ms_data, time_ms_fwd, time_ms_bwd, time_ms_model = ms(toc_data - tic), ms(toc_fwd - toc_data), ms(toc_bwd - toc_fwd), ms(toc_bwd - toc_data)	
			time_ms_avg = moving_avg(time_ms_avg, time_ms_data + time_ms_model, max = 10_000)
			print(f'{args.experiment_id} | epoch: {epoch:02d} iter: [{sampler.batch_idx: >6d} / {len(train_data_loader)} {iteration: >6d}] ent: <{entropy_avg:.2f}> loss: {loss_cur:.2f} <{loss_avg:.2f}> time: ({"x".join(map(str, input_.shape))}) {time_ms_data:.2f}+{time_ms_fwd:4.0f}+{time_ms_bwd:4.0f} <{time_ms_avg:.0f}> | lr: {lr:.5f}')
			iteration += 1

			if iteration > 0 and iteration % args.val_iteration_interval == 0:
				evaluate_model(val_data_loaders, epoch, iteration)

			if iteration and args.iterations and iteration >= args.iterations:
				sys.exit(0)

			if iteration % args.log_iteration_interval == 0:
				tensorboard.add_scalars(f'datasets/{train_dataset_name}', dict(loss_avg = loss_avg, lr_avg_x1e4 = lr_avg * 1e4), iteration)
				for param_name, param in model.module.named_parameters():
					if param.grad is None:
						continue
					tag = 'params/' + param_name.replace('.', '/')
					norm, grad_norm = param.norm(), param.grad.norm()
					ratio = grad_norm / (1e-9 + norm)
					tensorboard.add_scalars(tag, dict(norm = float(norm), grad_norm = float(grad_norm), ratio = float(ratio)), iteration)
					if args.log_weight_distribution:
						tensorboard.add_histogram(tag, param, iteration)
						tensorboard.add_histogram(tag + '/grad', param.grad, iteration)

			tic = time.time()

		print('Epoch time', (time.time() - time_epoch_start) / 60, 'minutes')
		sampler.shuffle(epoch) # TODO: redo bucketed sampling, move to beginning of epoch
		evaluate_model(val_data_loaders, epoch, iteration)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--optimizer', choices = ['SGD', 'AdamW', 'NovoGrad', 'FusedNovoGrad'], default = 'SGD')
	parser.add_argument('--max-norm', type = float, default = 100)
	parser.add_argument('--lr', type = float, default = 1e-2)
	parser.add_argument('--weight-decay', type = float, default = 1e-3)
	parser.add_argument('--momentum', type = float, default = 0.9)
	parser.add_argument('--nesterov', action = 'store_true')
	parser.add_argument('--betas', nargs = '*', type = float, default = (0.9, 0.999))
	parser.add_argument('--scheduler', choices = ['MultiStepLR', 'PolynomialDecayLR', 'StepLR'], default = None)
	parser.add_argument('--decay-gamma', type = float, default = 0.1)
	parser.add_argument('--decay-milestones', nargs = '*', type = int, default = [25_000, 50_000], help = 'for MultiStepLR')
	parser.add_argument('--decay-power', type = float, default = 2.0, help = 'for PolynomialDecayLR')
	parser.add_argument('--decay-lr', type = float, default = 1e-5, help = 'for PolynomialDecayLR')
	parser.add_argument('--decay-epochs', type = int, default = 5, help = 'for PolynomialDecayLR')
	parser.add_argument('--decay-step-size', type = int, default = 10_000, help = 'for PolynomialDecayLR')
	parser.add_argument('--fp16', choices = ['O0', 'O1', 'O2', 'O3'], default = None)
	parser.add_argument('--fp16-keep-batchnorm-fp32', default = None, action = 'store_true')
	parser.add_argument('--epochs', type = int, default = 5)
	parser.add_argument('--iterations', type = int, default = None)
	parser.add_argument('--train-data-path', nargs = '*', default = [])
	parser.add_argument('--train-data-mixing', type = float, nargs = '*')
	parser.add_argument('--val-data-path', nargs = '+')
	parser.add_argument('--num-workers', type = int, default = 32)
	parser.add_argument('--train-batch-size', type = int, default = 64)
	parser.add_argument('--val-batch-size', type = int, default = 64)
	parser.add_argument('--device', default = 'cuda', choices = ['cuda', 'cpu'])
	parser.add_argument('--checkpoint', nargs = '*', default = [], help = 'simple checkpoint weight averaging, for advanced weight averaging see Stochastic Weight Averaging')
	parser.add_argument('--checkpoint-skip', action = 'store_true')
	parser.add_argument('--experiments-dir', default = 'data/experiments')
	parser.add_argument('--experiment-dir', default = '{experiments_dir}/{experiment_id}')
	parser.add_argument('--checkpoint-format', default = 'checkpoint_epoch{epoch:02d}_iter{iteration:07d}.pt')
	parser.add_argument('--val-transcripts-format', default = 'data/transcripts_{val_dataset_name}_{decoder}.json', help = 'save transcripts at validation')
	parser.add_argument('--train-transcripts-format', default = 'transcripts_{val_dataset_name}_epoch{epoch:02d}_iter{iteration:07d}.json')
	parser.add_argument('--logits', nargs = '?', const = 'data/logits_{val_dataset_name}.pt', help = 'save logits at validation')
	parser.add_argument('--args', default = 'args.json', help = 'save experiment arguments to the experiment dir')
	parser.add_argument('--model', default = 'JasperNetBig')
	parser.add_argument('--seed', type = int, default = 1, help = 'reset to this random seed value in the main thread and data loader threads')
	parser.add_argument('--cudnn', default = 'benchmark', help = 'enables cudnn benchmark mode')
	parser.add_argument('--experiment-id', default = '{model}_{optimizer}_lr{lr:.0e}_wd{weight_decay:.0e}_bs{train_batch_size}_{train_waveform_transform}_{train_feature_transform}_{bpe}_{experiment_name}')
	parser.add_argument('--experiment-name', '--name', default = '')
	parser.add_argument('--comment', default = '')
	parser.add_argument('--dry', action = 'store_true')
	parser.add_argument('--lang', default = 'ru')
	parser.add_argument('--val-waveform-transform-debug-dir', help = 'debug dir for augmented audio at validation')
	parser.add_argument('--val-waveform-transform', nargs = '*', default = [], help = 'waveform transforms are applied before frontend to raw audio waveform')
	parser.add_argument('--val-waveform-transform-prob', type = float, default = None, help = 'apply transform with given probability')
	parser.add_argument('--val-feature-transform', nargs = '*', default = [], help = 'feature aug transforms are applied after frontend')
	parser.add_argument('--val-feature-transform-prob', type = float, default = None)
	parser.add_argument('--train-waveform-transform', nargs = '*', default = [])
	parser.add_argument('--train-waveform-transform-prob', type = float, default = None)
	parser.add_argument('--train-feature-transform', nargs = '*', default = [])
	parser.add_argument('--train-feature-transform-prob', type = float, default = None)
	parser.add_argument('--train-batch-accumulate-iterations', type = int, default = 1, help = 'number of gradient accumulation steps')
	parser.add_argument('--val-iteration-interval', type = int, default = 2500)
	parser.add_argument('--log-iteration-interval', type = int, default = 100)
	parser.add_argument('--log-weight-distribution', action = 'store_true')
	parser.add_argument('--verbose', action = 'store_true', help = 'print all transcripts to console, at validation')
	parser.add_argument('--align', nargs = '*', default = None)
	parser.add_argument('--decoder', default = 'GreedyDecoder', choices = ['GreedyDecoder', 'BeamSearchDecoder'])
	parser.add_argument('--decoder-topk', type = int, default = 1, help = 'compute CER for many decoding hypothesis (oracle)')
	parser.add_argument('--beam-width', type = int, default = 5000)
	parser.add_argument('--beam-alpha', type = float, default = 0.3, help = 'weight for language model (prob? log-prob?, TODO: check in ctcdecode)')
	parser.add_argument('--beam-beta', type = float, default = 1.0, help = 'weight for decoded transcript length (TODO: check in ctcdecode)')
	parser.add_argument('--lm', help = 'path to KenLM language model for ctcdecode')
	parser.add_argument('--bpe', nargs = '*', help = 'path to SentencePiece subword model', default = [])
	parser.add_argument('--timeout', type = float, default = 0, help = 'crash after specified timeout spent in DataLoader')
	parser.add_argument('--max-duration', type = float, default = 10, help = 'in seconds; drop in DataLoader all utterances longer than this value')
	parser.add_argument('--exphtml', default = '../stt_results')
	parser.add_argument('--freeze', action = 'store_true', help = 'freeze earlier layers')
	parser.add_argument('--num-input-features', default = 64, help = 'num of mel-scale features produced by logfbank frontend')
	parser.add_argument('--sample-rate', type = int, default = 8_000, help = 'for frontend')
	parser.add_argument('--window-size', type = float, default = 0.02, help = 'for frontend, in seconds')
	parser.add_argument('--window-stride', type = float, default = 0.01, help = 'for frontend, in seconds')
	parser.add_argument('--window', default = 'hann_window', choices = ['hann_window', 'hamming_window'], help = 'for frontend')
	parser.add_argument('--onnx')
	parser.add_argument('--dropout', type = float, default = 0.2)
	parser.add_argument('--githttp')

	traineval(parser.parse_args())
