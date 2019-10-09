import os
import sys
import json
import math
import time
import shutil
import random
import argparse
import importlib
import torch
import torch.utils.data
import torch.utils.tensorboard
import torch.nn as nn
import torch.nn.functional as F
import apex
import dataset
import transforms
import decoders
import metrics
import models
import optimizers

def set_random_seed(seed):
	for set_random_seed in [random.seed, torch.manual_seed] + ([torch.cuda.manual_seed_all] if torch.cuda.is_available() else []):
		set_random_seed(seed)

def worker_init_fn(worker_idx):
	set_random_seed(worker_idx)

def traineval(args):
	checkpoint = torch.load(args.checkpoint, map_location = 'cpu') if args.checkpoint else {}
	args.experiment_id = args.experiment_id.format(model = args.model, train_batch_size = args.train_batch_size, optimizer = args.optimizer, lr = args.lr, weight_decay = args.weight_decay, time = time.strftime('%Y-%m-%d_%H-%M-%S'), experiment_name = args.experiment_name, bpe = 'bpe' if args.bpe else '', train_waveform_transform = f'aug{args.train_waveform_transform[0]}{args.train_waveform_transform_prob or ""}' if args.train_waveform_transform else '', train_feature_transform = f'aug{args.train_feature_transform[0]}{args.train_feature_transform_prob or ""}' if args.train_feature_transform else '').replace('e-0', 'e-').rstrip('_')
	if 'experiment_id' in checkpoint and not args.experiment_name:
		args.experiment_id = checkpoint['experiment_id']

	args.lang, args.model, args.num_input_features = checkpoint.get('lang', args.lang), checkpoint.get('model', args.model), checkpoint.get('num_input_features', args.num_input_features)
	args.experiment_dir = args.experiment_dir.format(experiments_dir = args.experiments_dir, experiment_id = args.experiment_id)
	if not args.train_data_path and checkpoint and 'args' in checkpoint:
		args.sample_rate, args.window, args.window_size, args.window_stride = list(map(checkpoint['args'].get, ['sample_rate', 'window', 'window_size', 'window_stride']))
	print('\n', 'Arguments:', args)
	print('\n', 'Experiment id:', args.experiment_id, '\n')
	if args.dry:
		return
	set_random_seed(args.seed)
	if args.cudnn == 'benchmark':
		torch.backends.cudnn.benchmark = True

	labels = dataset.Labels(importlib.import_module(args.lang), bpe = args.bpe)
	make_transform = lambda name_args, prob: None if not name_args else getattr(transforms, name_args[0])(*name_args[1:]) if prob is None else getattr(transforms, name_args[0])(prob, *name_args[1:]) if prob > 0 else None
	val_waveform_transform = make_transform(args.val_waveform_transform, args.val_waveform_transform_prob)
	val_feature_transform = make_transform(args.val_feature_transform, args.val_feature_transform_prob)
	if args.val_waveform_transform_debug_dir:
		args.val_waveform_transform_debug_dir = os.path.join(args.val_waveform_transform_debug_dir, str(val_waveform_transform) if isinstance(val_waveform_transform, transforms.RandomCompose) else val_waveform_transform.__class__.__name__)
		os.makedirs(args.val_waveform_transform_debug_dir, exist_ok = True)
	val_data_loaders = {os.path.basename(val_data_path) : torch.utils.data.DataLoader(dataset.AudioTextDataset(val_data_path, sample_rate = args.sample_rate, window_size = args.window_size, window_stride = args.window_stride, window = args.window, labels = labels, num_input_features = args.num_input_features, waveform_transform = val_waveform_transform, waveform_transform_debug_dir = args.val_waveform_transform_debug_dir, feature_transform = val_feature_transform, max_duration = 20), num_workers = args.num_workers, collate_fn = dataset.collate_fn, pin_memory = True, shuffle = False, batch_size = args.val_batch_size, worker_init_fn = worker_init_fn, timeout = args.timeout) for val_data_path in args.val_data_path}
	model = getattr(models, args.model)(num_classes = len(labels), num_input_features = args.num_input_features, dropout = args.dropout)
	criterion = nn.CTCLoss(blank = labels.blank_idx, reduction = 'none').to(args.device)
	decoder = decoders.GreedyDecoder(labels) if args.decoder == 'GreedyDecoder' else decoders.BeamSearchDecoder(labels, lm_path = args.lm, beam_width = args.beam_width, beam_alpha = args.beam_alpha, beam_beta = args.beam_beta, num_workers = args.num_workers, topk = args.decoder_topk)

	def evaluate_model(epoch = None, iteration = None, batch_idx = None):
		training = epoch is not None and iteration is not None
		model.eval()
		for val_dataset_name, val_data_loader in val_data_loaders.items():
			features_, logits_, ref_tra_ = [], [], []
			for batch_idx_, (inputs, targets, input_lengths_fraction, target_lengths, filenames, dataset_names) in enumerate(val_data_loader):
				inputs, targets, input_lengths_fraction, target_lengths = inputs.to(args.device), targets.to(args.device), input_lengths_fraction.to(args.device), target_lengths.to(args.device)
				#inputs = models.logfbank(inputs, train_dataset.sample_rate, train_dataset.window_size, train_dataset.window_stride, train_dataset.window, train_dataset.num_input_features, normalize = train_dataset.normalize_features)
				with torch.no_grad():
					logits, output_lengths = model(inputs, input_lengths_fraction)
					log_probs = F.log_softmax(logits, dim = 1)
					loss = criterion(log_probs.permute(2, 0, 1), targets, output_lengths, target_lengths) / target_lengths
				entropy = models.entropy(log_probs, output_lengths, dim = 1)
				decoded_strings = labels.idx2str(decoder.decode(F.log_softmax(logits, dim = 1), output_lengths.tolist()))
				target_strings = labels.idx2str(targets, lengths = target_lengths)
				for k, (audio_path, transcript, reference, entropy, loss) in enumerate(zip(filenames, decoded_strings, target_strings, entropy, loss)):
					if isinstance(transcript, list):
						transcript = min((metrics.cer(t, reference), t) for t in transcript)[1]
					cer, wer = metrics.cer(transcript, reference), metrics.wer(transcript, reference) 
					transcript_, reference_ = metrics.align(transcript, reference, prepend_space_to_reference = True) if args.align else (transcript, reference)
					if args.verbose:
						print(iteration, ':', batch_idx_ * len(inputs) + k, '/', len(val_data_loader) * len(inputs))
						print(f'{val_dataset_name} REF: {reference_}')
						print(f'{val_dataset_name} HYP: {transcript_}')
						print(f'{val_dataset_name} WER: {wer:.02%} | CER: {cer:.02%}\n')
					ref_tra_.append(dict(reference = reference_, transcript = transcript_, audio_path = audio_path, filename = os.path.basename(audio_path), cer = cer, wer = wer, loss = float(loss), entropy = float(entropy)))
					if not training and args.logits:
						logits_.extend(logits.cpu())
						features_.extend(features.cpu())
			cer_avg, wer_avg, loss_avg, entropy_avg = [float(torch.tensor([x[k] for x in ref_tra_ if not math.isinf(x[k]) and not math.isnan(x[k])]).mean()) for k in ['cer', 'wer', 'loss', 'entropy']]
			transcripts_path = os.path.join(args.experiment_dir, f'transcripts_{val_dataset_name}_epoch{epoch:02d}_iter{iteration:07d}.json') if training else args.transcripts.format(val_dataset_name = val_dataset_name)
			print(f'{args.experiment_id} {val_dataset_name}', f'| epoch {epoch} iter {iteration}' if training else '', f'| {transcripts_path} | Entropy: {entropy_avg:.02f} Loss: {loss_avg:.02f} | WER:  {wer_avg:.02%} CER: {cer_avg:.02%}\n')

			with open(transcripts_path, 'w') as f:
				json.dump(list(sorted(ref_tra_, key = lambda ref_tra: ref_tra['cer'], reverse = True)), f, ensure_ascii = False, indent = 2, sort_keys = True)
			if training:
				tensorboard.add_scalars('datasets/' + val_dataset_name, dict(wer_avg = wer_avg * 100.0, cer_avg = cer_avg * 100.0, loss_avg = loss_avg), iteration) 
				tensorboard.flush()
			elif args.logits:
				torch.save(list(sorted([ref_tra.update(dict(logits = logits, features = features)) or ref_tra for features, logits, ref_tra in zip(features_, logits_, ref_tra_)], key = lambda j: j['cer'], reverse = True)), args.logits.format(val_dataset_name = val_dataset_name))
		if training and not args.checkpoint_skip:
			#TODO: amp.state_dict()
			optimizer_state_dict = None # optimizer.state_dict()
			torch.save(dict(model = model.module.__class__.__name__, model_state_dict = model.module.state_dict(), optimizer_state_dict = optimizer_state_dict, scheduler_state_dict = scheduler.state_dict(), sampler_state_dict = sampler.state_dict(batch_idx), epoch = epoch, iteration = iteration, args = vars(args), experiment_id = args.experiment_id, lang = args.lang, num_input_features = args.num_input_features, time = time.time()), os.path.join(args.experiment_dir, f'checkpoint_epoch{epoch:02d}_iter{iteration:07d}.pt'))
		if iteration and args.iterations and iteration >= args.iterations:
			sys.exit(0)

	print(' Model capacity:', int(models.compute_capacity(model) / 1e6), 'million parameters\n')

	if checkpoint:
		model.load_state_dict(checkpoint['model_state_dict'])
	model.to(args.device)

	if not args.train_data_path:
		if args.fp16:
			model = apex.amp.initialize(model, opt_level = args.fp16, keep_batchnorm_fp32 = args.fp16_keep_batchnorm_fp32)
		model = torch.nn.DataParallel(model)
		return evaluate_model()

	train_waveform_transform = make_transform(args.train_waveform_transform, args.train_waveform_transform_prob)
	train_feature_transform = make_transform(args.train_feature_transform, args.train_feature_transform_prob)
	train_dataset = dataset.AudioTextDataset(args.train_data_path, sample_rate = args.sample_rate, window_size = args.window_size, window_stride = args.window_stride, window = args.window, labels = labels, num_input_features = args.num_input_features, waveform_transform = train_waveform_transform, feature_transform = train_feature_transform, max_duration = args.max_duration)
	train_dataset_name = '_'.join(map(os.path.basename, args.train_data_path))
	sampler = dataset.BucketingSampler(train_dataset, batch_size = args.train_batch_size, mixing = args.train_data_mixing)
	train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers = args.num_workers, collate_fn = dataset.collate_fn, pin_memory = True, batch_sampler = sampler, worker_init_fn = worker_init_fn, timeout = args.timeout)
	optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay, nesterov = args.nesterov) if args.optimizer == 'SGD' else torch.optim.AdamW(model.parameters(), lr = args.lr, betas = args.betas, weight_decay = args.weight_decay) if args.optimizer == 'AdamW' else optimizers.NovoGrad(model.parameters(), lr = args.lr, betas = args.betas, weight_decay = args.weight_decay) if args.optimizer == 'NovoGrad' else apex.optimizers.FusedNovoGrad(model.parameters(), lr = args.lr, betas = args.betas, weight_decay = args.weight_decay) if args.optimizer == 'FusedNovoGrad' else None
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma = args.decay_gamma, milestones = args.decay_milestones) if args.scheduler == 'MultiStepLR' else optimizers.PolynomialDecayLR(optimizer, power = args.decay_power, decay_steps = len(train_data_loader) * args.decay_epochs, end_lr = args.decay_lr) if args.scheduler == 'PolynomialDecayLR' else torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.decay_step_size, gamma = args.decay_gamma) if args.scheduler == 'StepLR' else optimizers.NoopLR(optimizer) 
	epoch, iteration = 0, 0
	if checkpoint:
		#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		iteration = 1 + checkpoint.get('iteration', int(args.checkpoint[args.checkpoint.find('iter') + 4: -3]))
		tolist_ = lambda x: x if isinstance(x, list) else [x]
		if args.train_data_path == tolist_(checkpoint['args']['train_data_path']):
			sampler.load_state_dict(checkpoint['sampler_state_dict'])
			scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

			if sampler.batch_idx + 1 == len(sampler):
				sampler.shuffle(epoch = sampler.epoch + 1, batch_idx = 0)
			else:
				sampler.shuffle(epoch = sampler.epoch, batch_idx = sampler.batch_idx + 1)
				#sampler.batch_idx += 1
		else:
			sampler.shuffle(epoch = sampler.epoch + 1)

	if args.fp16:
		model, optimizer = apex.amp.initialize(model, optimizer, opt_level = args.fp16, keep_batchnorm_fp32 = args.fp16_keep_batchnorm_fp32)
	model = torch.nn.DataParallel(model)

	os.makedirs(args.experiment_dir, exist_ok = True)
	tensorboard_dir = os.path.join(args.experiment_dir, 'tensorboard')
	if checkpoint and args.experiment_name:
		tensorboard_dir_checkpoint = os.path.join(os.path.dirname(args.checkpoint), 'tensorboard')
		if os.path.exists(tensorboard_dir_checkpoint) and not os.path.exists(tensorboard_dir):
			shutil.copytree(tensorboard_dir_checkpoint, tensorboard_dir)
	tensorboard = torch.utils.tensorboard.SummaryWriter(tensorboard_dir)
	with open(os.path.join(args.experiment_dir, args.args), 'w') as f:
		json.dump(vars(args), f, sort_keys = True, ensure_ascii = False, indent = 2)

	tic, toc_fwd, toc_bwd = time.time(), time.time(), time.time()
	loss_avg, entropy_avg, time_ms_avg, lr_avg = 0.0, 0.0, 0.0, 0.0
	moving_avg = lambda avg, x, max = 0, K = 50: (1. / K) * min(x, max) + (1 - 1./K) * avg
	for epoch in range(sampler.epoch, args.epochs):
		model.train()
		time_epoch_start = time.time()
		for batch_idx, (inputs, targets, input_lengths_fraction, target_lengths, filenames, dataset_names) in enumerate(train_data_loader, start = sampler.batch_idx):
			toc = time.time()
			inputs, targets, input_lengths_fraction, target_lengths = inputs.to(args.device), targets.to(args.device), input_lengths_fraction.to(args.device), target_lengths.to(args.device)
			#inputs = models.logfbank(inputs, train_dataset.sample_rate, train_dataset.window_size, train_dataset.window_stride, train_dataset.window, train_dataset.num_input_features, normalize = train_dataset.normalize_features)
			with torch.enable_grad():
				logits, output_lengths = model(inputs, input_lengths_fraction)
				log_probs = F.log_softmax(logits, dim = 1)
				loss_ = criterion(log_probs.permute(2, 0, 1), targets, output_lengths, target_lengths)# / target_lengths
				#loss_, loss = loss_.mean(), (loss_ * target_lengths).mean()
				loss_ = loss = loss_.mean()
				entropy = models.entropy(log_probs, output_lengths, dim = 1).mean()
				if not (torch.isinf(loss) or torch.isnan(loss)):
					toc_fwd = time.time()
					optimizer.zero_grad()
					if args.fp16:
						with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
							scaled_loss.backward()
					else:
						loss.backward()
					torch.nn.utils.clip_grad_norm_(apex.amp.master_params(optimizer) if args.fp16 else model.parameters(), args.max_norm)
					optimizer.step()
					loss_avg = moving_avg(loss_avg, float(loss_), max = 1000)
					entropy_avg = moving_avg(entropy_avg, float(entropy), max = 4)
					toc_bwd = time.time()

			ms = lambda sec: sec * 1000
			time_ms_data, time_ms_fwd, time_ms_bwd, time_ms_model = ms(toc - tic), ms(toc_fwd - toc), ms(toc_bwd - toc_fwd), ms(time.time() - toc)	
			time_ms_avg = moving_avg(time_ms_avg, time_ms_model, max = 10000)
			lr = scheduler.get_lr()[0]; lr_avg = moving_avg(lr_avg, lr, max = 1)
			print(f'{args.experiment_id} | epoch: {epoch:02d} iter: [{batch_idx: >6d} / {len(train_data_loader)} {iteration: >6d}] ent: <{entropy_avg:.2f}>  loss: {float(loss_): 7.2f} <{loss_avg: 7.2f}> time: ({inputs.shape[-1]: >3d}) {time_ms_data:.2f}+{time_ms_fwd:4.0f}+{time_ms_bwd:4.0f} <{time_ms_avg:4.0f}> | lr: {lr:.5f}')
			tic = time.time()
			scheduler.step()
			iteration += 1

			if args.val_iteration_interval is not None and iteration > 0 and iteration % args.val_iteration_interval == 0:
				evaluate_model(epoch, iteration, batch_idx)
				model.train()

			if iteration % args.log_iteration_interval == 0:
				tensorboard.add_scalars('datasets/' + train_dataset_name, dict(loss_avg = loss_avg, lr_avg_x1e4 = lr_avg * 1e4), iteration)
				for param_name, param in model.module.named_parameters():
					tag = 'params/' + param_name.replace('.', '/')
					norm, grad_norm = param.norm(), param.grad.norm()
					ratio = grad_norm / (1e-9 + norm)
					tensorboard.add_scalars(tag, dict(norm = float(norm), grad_norm = float(grad_norm), ratio = float(ratio)), iteration)
					if args.log_weight_distribution:
						tensorboard.add_histogram(tag, param, iteration)
						tensorboard.add_histogram(tag + '/grad', param.grad, iteration)

		sampler.shuffle(epoch)
		evaluate_model(epoch, iteration)
		print('Epoch time', (time.time() - time_epoch_start) / 60, 'minutes')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--optimizer', choices = ['SGD', 'AdamW', 'NovoGrad', 'FusedNovoGrad'], default = 'SGD')
	parser.add_argument('--max-norm', type = float, default = 100)
	parser.add_argument('--lr', type = float, default = 5e-3)
	parser.add_argument('--weight-decay', type = float, default = 1e-3)
	parser.add_argument('--momentum', type = float, default = 0.9)
	parser.add_argument('--nesterov', action = 'store_true')
	parser.add_argument('--betas', nargs = '*', type = float, default = (0.9, 0.999))
	parser.add_argument('--scheduler', choices = ['MultiStepLR', 'PolynomialDecayLR', 'StepLR'], default = None)
	parser.add_argument('--decay-gamma', type = float, default = 0.1)
	parser.add_argument('--decay-milestones', nargs = '*', type = int, default = [25_000, 50_000])
	parser.add_argument('--decay-power', type = float, default = 2.0)
	parser.add_argument('--decay-lr', type = float, default = 1e-5)
	parser.add_argument('--decay-epochs', type = int, default = 5)
	parser.add_argument('--decay-step-size', type = int, default = 10_000)
	parser.add_argument('--fp16', choices = ['O0', 'O1', 'O2', 'O3'], default = None)
	parser.add_argument('--fp16-keep-batchnorm-fp32', default = None, action = 'store_true')
	parser.add_argument('--epochs', type = int, default = 10)
	parser.add_argument('--iterations', type = int, default = None)
	parser.add_argument('--num-input-features', default = 64)
	parser.add_argument('--dropout', type = float, default = 0.2)
	parser.add_argument('--train-data-path', nargs = '*')
	parser.add_argument('--train-data-mixing', type = float, nargs = '*')
	parser.add_argument('--val-data-path', nargs = '+')
	parser.add_argument('--sample-rate', type = int, default = 8_000)
	parser.add_argument('--window-size', type = float, default = 0.02)
	parser.add_argument('--window-stride', type = float, default = 0.01)
	parser.add_argument('--window', default = 'hann_window', choices = ['hann_window', 'hamming_window'])
	parser.add_argument('--num-workers', type = int, default = 32)
	parser.add_argument('--train-batch-size', type = int, default = 64)
	parser.add_argument('--val-batch-size', type = int, default = 64)
	parser.add_argument('--device', default = 'cuda', choices = ['cuda', 'cpu'])
	parser.add_argument('--checkpoint')
	parser.add_argument('--checkpoint-skip', action = 'store_true')
	parser.add_argument('--experiments-dir', default = 'data/experiments')
	parser.add_argument('--experiment-dir', default = '{experiments_dir}/{experiment_id}')
	parser.add_argument('--transcripts', default = 'data/transcripts_{val_dataset_name}.json')
	parser.add_argument('--logits', nargs = '?', const = 'data/logits_{val_dataset_name}.pt')
	parser.add_argument('--args', default = 'args.json')
	parser.add_argument('--model', default = 'Wav2LetterRu')
	parser.add_argument('--seed', type = int, default = 1)
	parser.add_argument('--cudnn', default = 'benchmark')
	parser.add_argument('--experiment-id', default = '{model}_{optimizer}_lr{lr:.0e}_wd{weight_decay:.0e}_bs{train_batch_size}_{train_waveform_transform}_{train_feature_transform}_{experiment_name}_{bpe}')
	parser.add_argument('--experiment-name', '--name', default = '')
	parser.add_argument('--dry', action = 'store_true')
	parser.add_argument('--lang', default = 'ru')
	parser.add_argument('--val-waveform-transform-debug-dir')
	parser.add_argument('--val-waveform-transform', nargs = '*', default = [])
	parser.add_argument('--val-waveform-transform-prob', type = float, default = None)
	parser.add_argument('--val-feature-transform', nargs = '*', default = [])
	parser.add_argument('--val-feature-transform-prob', type = float, default = None)
	parser.add_argument('--train-waveform-transform', nargs = '*', default = [])
	parser.add_argument('--train-waveform-transform-prob', type = float, default = None)
	parser.add_argument('--train-feature-transform', nargs = '*', default = [])
	parser.add_argument('--train-feature-transform-prob', type = float, default = None)
	parser.add_argument('--val-iteration-interval', type = int, default = 2500)
	parser.add_argument('--log-iteration-interval', type = int, default = 100)
	parser.add_argument('--log-weight-distribution', action = 'store_true')
	parser.add_argument('--verbose', action = 'store_true')
	parser.add_argument('--align', action = 'store_true')
	parser.add_argument('--decoder', default = 'GreedyDecoder', choices = ['GreedyDecoder', 'BeamSearchDecoder'])
	parser.add_argument('--decoder-topk', type = int, default = 1)
	parser.add_argument('--beam-width', type = int, default = 5000)
	parser.add_argument('--beam-alpha', type = float, default = 0.3)
	parser.add_argument('--beam-beta', type = float, default = 1.0)
	parser.add_argument('--lm')
	parser.add_argument('--bpe')
	parser.add_argument('--timeout', type = float, default = 0)
	parser.add_argument('--max-duration', type = float, default = 10)
	
	traineval(parser.parse_args())
