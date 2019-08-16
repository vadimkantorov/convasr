import os
import gc
import json
import time
import random
import argparse
import importlib
import torch
import torch.utils.data
import torch.utils.tensorboard
import torch.nn as nn
import torch.nn.functional as F
import dataset
import transforms
import decoders
import models
import optimizers
import torchaudio
try:
	import apex
except:
	pass

def load_checkpoint(checkpoint_path, model, optimizer = None, sampler = None, scheduler = None):
	checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
	model.load_state_dict(checkpoint['model_state_dict'])
	if optimizer is not None:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	if sampler is not None:
		sampler.load_state_dict(checkpoint['sampler_state_dict'])
	if scheduler is not None:
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

def save_checkpoint(checkpoint_path, model, optimizer, sampler, scheduler, epoch, batch_idx):
	checkpoint = dict(model = model.__class__.__name__, model_state_dict = model.state_dict(), optimizer_state_dict = optimizer.state_dict(), scheduler_state_dict = scheduler.state_dict(), sampler_state_dict = sampler.state_dict(batch_idx), epoch = epoch)
	torch.save(checkpoint, checkpoint_path)

def set_random_seed(seed):
	for set_random_seed in [random.seed, torch.manual_seed] + ([torch.cuda.manual_seed_all] if torch.cuda.is_available() else []):
		set_random_seed(seed)

def traineval(args):
	print('\n', 'Arguments:', args)
	args.experiment_id = args.experiment_id.format(model = args.model, train_batch_size = args.train_batch_size, optimizer = args.optimizer, lr = args.lr, weight_decay = args.weight_decay, time = time.strftime('%Y-%m-%d_%H-%M-%S'), experiment_name = args.experiment_name, train_waveform_transform = f'aug{args.train_waveform_transform}{args.train_waveform_transform_prob or ""}' if args.train_waveform_transform_prob is None or args.train_waveform_transform_prob > 0 else '').replace('e-0', 'e-').rstrip('_')
	if args.train_data_path:
		print('\n', 'Experiment id:', args.experiment_id, '\n')
	if args.dry:
		return
	args.experiment_dir = args.experiment_dir.format(experiments_dir = args.experiments_dir, experiment_id = args.experiment_id)
	set_random_seed(args.seed)

	labels = dataset.Labels(importlib.import_module(args.lang))
	transform = lambda name, prob, args: None if not name else getattr(transforms, name)(*args) if prob is None else getattr(transforms, name)(prob, *args) if prob > 0 else None
	val_waveform_transform = transform(args.val_waveform_transform, args.val_waveform_transform_prob, args.val_waveform_transform_args)
	val_feature_transform = transform(args.val_feature_transform, args.val_feature_transform_prob, args.val_feature_transform_args)
	if args.val_waveform_transform_debug_dir:
		args.val_waveform_transform_debug_dir = os.path.join(args.val_waveform_transform_debug_dir, str(val_waveform_transform) if isinstance(val_waveform_transform, transforms.RandomCompose) else val_waveform_transform.__class__.__name__)
		os.makedirs(args.val_waveform_transform_debug_dir, exist_ok = True)
	val_data_loaders = {os.path.basename(val_data_path) : torch.utils.data.DataLoader(dataset.SpectrogramDataset(val_data_path, sample_rate = args.sample_rate, window_size = args.window_size, window_stride = args.window_stride, window = args.window, labels = labels, num_input_features = args.num_input_features, waveform_transform = val_waveform_transform, waveform_transform_debug_dir = args.val_waveform_transform_debug_dir, feature_transform = val_feature_transform), num_workers = args.num_workers, collate_fn = dataset.collate_fn, pin_memory = True, shuffle = False, batch_size = args.val_batch_size, worker_init_fn = set_random_seed) for val_data_path in args.val_data_path}
	model = getattr(models, args.model)(num_classes = len(labels), num_input_features = args.num_input_features)
	criterion = nn.CTCLoss(blank = labels.blank_idx, reduction = 'none').to(args.device)
	decoder = decoders.GreedyDecoder(labels) if args.decoder == 'GreedyDecoder' else decoders.BeamSearchDecoder(labels, lm_path = args.lm, beam_width = args.beam_width, beam_alpha = args.beam_alpha, beam_beta = args.beam_beta, num_workers = args.num_workers, topk = args.decoder_topk)

	def evaluate_model(epoch = None, batch_idx = None, iteration = None):
		training = epoch is not None and batch_idx is not None and iteration is not None
		model.eval()
		with torch.no_grad():
			for val_dataset_name, val_data_loader in val_data_loaders.items():
				logits_, ref_tra_, loss_ = [], [], []
				for batch_idx, (inputs, targets, filenames, input_percentages, target_lengths) in enumerate(val_data_loader):
					input_lengths = (input_percentages.cpu() * inputs.shape[-1]).int()
					output_lengths = models.compute_output_lengths(model, input_lengths)
					logits = model(inputs.to(args.device), input_lengths)
					log_probs = F.log_softmax(logits, dim = 1)
					loss = criterion(log_probs.permute(2, 0, 1), targets.to(args.device), output_lengths.to(args.device), target_lengths.to(args.device)).mean()
					loss_.append(float(loss))
					decoded_strings = labels.idx2str(decoder.decode(F.log_softmax(logits, dim = 1), output_lengths.tolist()))
					target_strings = labels.idx2str(dataset.unpack_targets(targets.tolist(), target_lengths.tolist()))
					for k, (transcript, reference) in enumerate(zip(decoded_strings, target_strings)):
						if isinstance(transcript, list):
							transcript = min((decoders.compute_cer(t, reference), t) for t in transcript)[1]
						cer, wer = decoders.compute_cer(transcript, reference), decoders.compute_wer(transcript, reference) 
						if args.verbose:
							print(batch_idx * len(inputs) + k, '/', len(val_data_loader) * len(inputs))
							print(f'{val_dataset_name} REF: {reference}')
							print(f'{val_dataset_name} HYP: {transcript}')
							print(f'{val_dataset_name} CER: {cer:.02%}  |  WER: {wer:.02%}')
							print()
						ref_tra_.append(dict(reference = reference, transcript = transcript, filename = filenames[k], cer = cer, wer = wer))
						if not training and args.logits:
							logits_.extend(logits.cpu())
				cer_avg, wer_avg = [float(torch.tensor([x[k] for x in ref_tra_]).mean()) for k in ['cer', 'wer']]
				loss_avg = float(torch.tensor(loss_).mean())
				print(f'{args.experiment_id} {val_dataset_name} | epoch {epoch} iter {iteration} | Loss: {loss_avg:.02f} | WER:  {wer_avg:.02%} CER: {cer_avg:.02%}')
				print()
				with open(os.path.join(args.experiment_dir, f'transcripts_{val_dataset_name}_epoch{epoch:02d}_iter{iteration:07d}.json') if training else args.transcripts.format(val_dataset_name = val_dataset_name), 'w') as f:
					json.dump(ref_tra_, f, ensure_ascii = False, indent = 2, sort_keys = True)
				if training:
					tensorboard.add_scalars(args.experiment_id, {val_dataset_name + '_wer_avg' : wer_avg * 100.0, val_dataset_name + '_cer_avg' : cer_avg * 100.0, val_dataset_name + '_loss_avg' : loss_avg}, iteration) 
				elif args.logits:
					torch.save(dict(logits = logits_, ref_tra = ref_tra_), args.logits.format(val_dataset_name = val_dataset_name))
		if training:
			save_checkpoint(os.path.join(args.experiment_dir, f'checkpoint_epoch{epoch:02d}_iter{iteration:07d}.pt'), model.module, optimizer, train_sampler, scheduler, epoch, batch_idx)
			model.train()

	if not args.train_data_path:
		load_checkpoint(args.checkpoint, model)
		model = torch.nn.DataParallel(model).to(args.device)
		return evaluate_model()

	train_waveform_transform = transform(args.train_waveform_transform, args.train_waveform_transform_prob, args.train_waveform_transform_args)
	train_dataset = dataset.SpectrogramDataset(args.train_data_path, sample_rate = args.sample_rate, window_size = args.window_size, window_stride = args.window_stride, window = args.window, labels = labels, num_input_features = args.num_input_features, waveform_transform = train_waveform_transform)
	train_dataset_name = os.path.basename(args.train_data_path)
	train_sampler = dataset.BucketingSampler(train_dataset, batch_size=args.train_batch_size)
	train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers = args.num_workers, collate_fn = dataset.collate_fn, pin_memory = True, batch_sampler = train_sampler)
	optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay, nesterov = args.nesterov) if args.optimizer == 'SGD' else torch.optim.AdamW(model.parameters(), lr = args.lr, betas = args.betas, weight_decay = args.weight_decay) if args.optimizer == 'AdamW' else optimizers.NovoGrad(model.parameters(), lr = args.lr, betas = args.betas, weight_decay = args.weight_decay) if args.optimizer == 'NovoGrad' else None
	if args.fp16:
		model, optimizer = apex.amp.initialize(model, optimizer, opt_level = args.fp16, keep_batchnorm_fp32 = args.fp16_keep_batchnorm_fp32)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma = args.decay_gamma, milestones = args.decay_milestones) if args.scheduler == 'MultiStepLR' else optimizers.PolynomialDecayLR(optimizer, power = args.decay_power, decay_steps = len(train_data_loader) * args.decay_epochs, end_lr = args.decay_lr) if args.scheduler == 'PolynomialDecayLR' else torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.decay_step_size, gamma = args.decay_gamma) if args.scheduler == 'StepLR' else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)
	if args.checkpoint:
		models.load_checkpoint(args.checkpoint, model, optimizer, train_sampler, scheduler)
	model = torch.nn.DataParallel(model).to(args.device)
	os.makedirs(args.experiment_dir, exist_ok = True)
	tensorboard = torch.utils.tensorboard.SummaryWriter(os.path.join(args.experiment_dir, 'tensorboard'))
	with open(os.path.join(args.experiment_dir, args.args), 'w') as f:
		json.dump(vars(args), f, sort_keys = True, ensure_ascii = False, indent = 2)

	tic = time.time()
	iteration, loss_avg, time_ms_avg, lr_avg = 0, 0.0, 0.0, 0.0
	moving_avg = lambda avg, x, max = 0, K = 50: (1. / K) * min(x, max) + (1 - 1./K) * avg
	for epoch in range(args.epochs):
		model.train()
		for batch_idx, (inputs, targets, filenames, input_percentages, target_lengths) in enumerate(train_data_loader):
			toc = time.time()
			input_lengths = (input_percentages.cpu() * inputs.shape[-1]).int()
			output_lengths = models.compute_output_lengths(model, input_lengths)
			logits = model(inputs.to(args.device), input_lengths)
			log_probs = F.log_softmax(logits, dim = 1)
			loss = criterion(log_probs.permute(2, 0, 1), targets.to(args.device), output_lengths.to(args.device), target_lengths.to(args.device)).mean()
			if not (torch.isinf(loss) or torch.isnan(loss)):
				optimizer.zero_grad()
				if args.fp16:
					with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
						scaled_loss.backward()
				else:
					loss.backward()
				torch.nn.utils.clip_grad_norm_(apex.amp.master_params(optimizer) if args.fp16 else model.parameters(), args.max_norm)
				optimizer.step()
				loss_avg = moving_avg(loss_avg, float(loss), max = 1000)

			time_ms_data, time_ms_model = (toc - tic) * 1000, (time.time() - toc) * 1000
			time_ms_avg = moving_avg(time_ms_avg, time_ms_model, max = 10000)
			lr = scheduler.get_lr()[0]; lr_avg = moving_avg(lr_avg, lr, max = 1)
			print(f'{args.experiment_id} | epoch: {epoch:02d} iter: [{batch_idx: >6d} / {len(train_data_loader)} {iteration: >9d}] loss: {float(loss): 7.2f} <{loss_avg: 7.2f}> time: {time_ms_model:6.0f} <{time_ms_avg:4.0f}> +{time_ms_data:.2f} ms  | lr: {lr}')#<{lr:.0e}>')
			tic = time.time()
			scheduler.step()
			iteration += 1

			if args.val_iteration_interval is not None and iteration > 0 and iteration % args.val_iteration_interval == 0:
				evaluate_model(epoch, batch_idx, iteration)

			if iteration % args.log_iteration_interval == 0:
				tensorboard.add_scalars(args.experiment_id, {train_dataset_name + '_loss_avg' : loss_avg, train_dataset_name + '_lr_avg_x10K' : lr_avg * 1e4}, iteration)
				for param_name, param in model.module.named_parameters():
					tag = args.experiment_id + '_params/' + param_name.replace('.', '/')
					norm, grad_norm = param.norm(), param.grad.norm()
					ratio = grad_norm / (1e-9 + norm)
					tensorboard.add_scalars(tag, dict(norm = float(norm), grad_norm = float(grad_norm), ratio = float(ratio)), iteration)
					if args.log_weight_distribution:
						tensorboard.add_histogram(tag, param, iteration)
						tensorboard.add_histogram(tag + '/grad', param.grad, iteration)

		evaluate_model(epoch, batch_idx, iteration)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--optimizer', choices = ['SGD', 'AdamW', 'NovoGrad'], default = 'SGD')
	parser.add_argument('--max-norm', type = float, default = 100)
	parser.add_argument('--lr', type = float, default = 5e-3)
	parser.add_argument('--weight-decay', type = float, default = 1e-5)
	parser.add_argument('--momentum', type = float, default = 0.9)
	parser.add_argument('--nesterov', action = 'store_true')
	parser.add_argument('--betas', nargs = '*', type = float, default = (0.9, 0.999))
	parser.add_argument('--scheduler', choices = ['MultiStepLR', 'PolynomialDecayLR', 'StepLR'], default = None)
	parser.add_argument('--decay-gamma', type = float, default = 0.1)
	parser.add_argument('--decay-milestones', nargs = '*', type = int, default = [25000])
	parser.add_argument('--decay-power', type = float, default = 2.0)
	parser.add_argument('--decay-lr', type = float, default = 1e-5)
	parser.add_argument('--decay-epochs', type = int, default = 5)
	parser.add_argument('--decay-step-size', type = int, default = 10000)
	parser.add_argument('--fp16', choices = ['O0', 'O1', 'O2', 'O3'], default = None)
	parser.add_argument('--fp16-keep-batchnorm-fp32', default = None, action = 'store_true')
	parser.add_argument('--epochs', type = int, default = 10)
	parser.add_argument('--num-input-features', default = 64)
	parser.add_argument('--train-data-path')
	parser.add_argument('--val-data-path', nargs = '+')
	parser.add_argument('--sample-rate', type = int, default = 16000)
	parser.add_argument('--window-size', type = float, default = 0.02)
	parser.add_argument('--window-stride', type = float, default = 0.01)
	parser.add_argument('--window', default = 'hann_window', choices = ['hann_window', 'hamming_window'])
	parser.add_argument('--num-workers', type = int, default = 20)
	parser.add_argument('--train-batch-size', type = int, default = 64)
	parser.add_argument('--val-batch-size', type = int, default = 64)
	parser.add_argument('--device', default = 'cuda', choices = ['cuda', 'cpu'])
	parser.add_argument('--checkpoint')
	parser.add_argument('--experiments-dir', default = 'data/experiments')
	parser.add_argument('--experiment-dir', default = '{experiments_dir}/{experiment_id}')
	parser.add_argument('--transcripts', default = 'data/transcripts_{val_dataset_name}.json')
	parser.add_argument('--logits', nargs = '?', const = 'data/logits_{val_dataset_name}.pt')
	parser.add_argument('--args', default = 'args.json')
	parser.add_argument('--model', default = 'Wav2LetterRu')
	parser.add_argument('--seed', type = int, default = 1)
	parser.add_argument('--experiment-id', default = '{model}_{optimizer}_lr{lr:.0e}_wd{weight_decay:.0e}_bs{train_batch_size}_{train_waveform_transform}_{experiment_name}')
	parser.add_argument('--experiment-name', '--name', default = '')
	parser.add_argument('--dry', action = 'store_true')
	parser.add_argument('--lang', default = 'ru')
	parser.add_argument('--val-waveform-transform-debug-dir')
	parser.add_argument('--val-waveform-transform')
	parser.add_argument('--val-waveform-transform-prob', type = float, default = None)
	parser.add_argument('--val-waveform-transform-args', nargs = '*', default = [])
	parser.add_argument('--val-feature-transform')
	parser.add_argument('--val-feature-transform-prob', type = float, default = None)
	parser.add_argument('--val-feature-transform-args', nargs = '*', default = [])
	parser.add_argument('--train-waveform-transform')
	parser.add_argument('--train-waveform-transform-prob', type = float, default = None)
	parser.add_argument('--train-waveform-transform-args', nargs = '*', default = [])
	parser.add_argument('--train-feature-transform')
	parser.add_argument('--val-iteration-interval', type = int, default = None)
	parser.add_argument('--log-iteration-interval', type = int, default = 100)
	parser.add_argument('--log-weight-distribution', action = 'store_true')
	parser.add_argument('--augment', action = 'store_true')
	parser.add_argument('--verbose', action = 'store_true')
	parser.add_argument('--decoder', default = 'GreedyDecoder', choices = ['GreedyDecoder', 'BeamSearchDecoder'])
	parser.add_argument('--decoder-topk', type = int, default = 1)
	parser.add_argument('--beam-width', type = int, default = 5000)
	parser.add_argument('--beam-alpha', type = float, default = 0.3)
	parser.add_argument('--beam-beta', type = float, default = 1.0)
	parser.add_argument('--lm')
	
	traineval(parser.parse_args())
