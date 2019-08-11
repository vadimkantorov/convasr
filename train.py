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
try:
	import apex
except:
	pass

def set_random_seed(seed):
	for set_random_seed in [random.seed, torch.manual_seed] + ([torch.cuda.manual_seed_all] if torch.cuda.is_available() else []):
		set_random_seed(seed)

def traintest(args):
	print('\n', 'Arguments:', args)
	args.id = args.id.format(model = args.model, train_batch_size = args.train_batch_size, optimizer = args.optimizer, lr = args.lr, weight_decay = args.weight_decay, time = time.strftime('%Y-%m-%d_%H-%M-%S'), name = args.name).replace('e-0', 'e-').rstrip('_')
	print('\n', 'Experiment id:', args.id, '\n')
	if args.dry:
		return
	args.experiment_dir = args.experiment_dir.format(experiments_dir = args.experiments_dir, id = args.id)
	set_random_seed(args.seed)

	labels = dataset.Labels(importlib.import_module(args.lang))
	val_waveform_transforms = eval(f'[{args.val_waveform_transforms}]', vars(transforms))
	val_data_loaders = {os.path.basename(val_data_path) : torch.utils.data.DataLoader(dataset.SpectrogramDataset(val_data_path, sample_rate = args.sample_rate, window_size = args.window_size, window_stride = args.window_stride, window = args.window, labels = labels, num_input_features = args.num_input_features, waveform_transforms = val_waveform_transforms), num_workers = args.num_workers, collate_fn = dataset.collate_fn, pin_memory = True, shuffle = False, batch_size = args.val_batch_size, worker_init_fn = set_random_seed) for val_data_path in args.val_data_path}
	model = getattr(models, args.model)(num_classes = len(labels), num_input_features = args.num_input_features)
	criterion = nn.CTCLoss(blank = labels.blank_idx, reduction = 'sum').to(args.device)
	model = torch.nn.DataParallel(model).to(args.device)
	decoder = decoders.GreedyDecoder(labels)

	def evaluate_model(epoch = None, batch_idx = None, iteration = None):
		training = epoch is not None and batch_idx is not None and iteration is not None
		model.eval()
		with torch.no_grad():
			for val_dataset_name, val_data_loader in val_data_loaders.items():
				logits_, ref_tra_, loss_ = [], [], []
				for i, (inputs, targets, filenames, input_percentages, target_lengths) in enumerate(val_data_loader):
					input_lengths = (input_percentages.cpu() * inputs.shape[-1]).int()
					logits = model(inputs.to(args.device), input_lengths)
					output_lengths = models.compute_output_lengths(model, input_lengths)
					loss = criterion(F.log_softmax(logits.permute(2, 0, 1), dim = -1), targets.to(args.device), output_lengths.to(args.device), target_lengths.to(args.device)) / len(inputs)
					loss_.append(float(loss))
					decoded_strings = labels.idx2str(decoder.decode(F.softmax(logits, dim = 1), output_lengths.tolist()))
					target_strings = labels.idx2str(dataset.unpack_targets(targets.tolist(), target_lengths.tolist()))
					for k, (transcript, reference) in enumerate(zip(decoded_strings, target_strings)):
						transcript, reference = transcript.strip(), reference.strip()
						wer_ref = len(reference.split()) or 1
						cer_ref = len(reference.replace(' ','')) or 1
						wer, cer = (decoders.compute_wer(transcript, reference), decoders.compute_cer(transcript, reference)) if reference != transcript else (0, 0)
						if args.verbose:
							print(val_dataset_name, 'REF: ', reference)
							print(val_dataset_name, 'HYP: ', transcript)
							print()
						wer, cer = wer / wer_ref,  cer / cer_ref
						ref_tra_.append(dict(reference = reference, transcript = transcript, filename = filenames[k], cer = cer, wer = wer))
						logits_.extend(logits)
				cer_avg = float(torch.tensor([x['cer'] for x in ref_tra_]).mean())
				wer_avg = float(torch.tensor([x['wer'] for x in ref_tra_]).mean())
				loss_avg = float(torch.tensor(loss_).mean())
				print(f'{args.id} {val_dataset_name} | epoch {epoch} iter {iteration} | Loss: {loss_avg:.02f} | WER:  {wer_avg:.02%} CER: {cer_avg:.02%}')
				print()
				with open(os.path.join(args.experiment_dir, f'transcripts_{val_dataset_name}_epoch{epoch:02d}_iter{iteration:07d}.json') if training else args.transcripts.format(val_dataset_name = val_dataset_name), 'w') as f:
					json.dump(ref_tra_, f, ensure_ascii = False, indent = 2, sort_keys = True)
				torch.save(dict(logits = logits_, ref_tra = ref_tra_), os.path.join(args.experiment_dir, f'logits_{val_dataset_name}_epoch{epoch:02d}_iter{iteration:07d}.pt') if training else args.logits.format(val_dataset_name = val_dataset_name))
		model.train()
		if training:
			tensorboard.add_scalars(args.id, {val_dataset_name + '_wer_avg' : wer_avg * 100.0, val_dataset_name + '_cer_avg' : cer_avg * 100.0, val_dataset_name + '_loss_avg' : loss_avg}, iteration) 
			models.save_checkpoint(os.path.join(args.experiment_dir, f'checkpoint_epoch{epoch:02d}_iter{iteration:07d}.pt'), model.module, optimizer, train_sampler, epoch, batch_idx)

	if not args.train_data_path:
		models.load_checkpoint(args.checkpoint, model)
		return evaluate_model()

	train_waveform_transforms = eval(f'[{args.train_waveform_transforms}]', vars(transforms))
	train_dataset = dataset.SpectrogramDataset(args.train_data_path, sample_rate = args.sample_rate, window_size = args.window_size, window_stride = args.window_stride, window = args.window, labels = labels, num_input_features = args.num_input_features, waveform_transforms = train_waveform_transforms)
	train_dataset_name = os.path.basename(args.train_data_path)
	train_sampler = dataset.BucketingSampler(train_dataset, batch_size=args.train_batch_size)
	train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers = args.num_workers, collate_fn = dataset.collate_fn, pin_memory = True, batch_sampler = train_sampler)
	optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay, nesterov = args.nesterov) if args.optimizer == 'SGD' else torch.optim.AdamW(model.parameters(), lr = args.lr, betas = args.betas, weight_decay = args.weight_decay) if args.optimizer == 'AdamW' else optimizers.NovoGrad(model.parameters(), lr = args.lr, betas = args.betas, weight_decay = args.weight_decay) if args.optimizer == 'NovoGrad' else None
	if args.fp16:
		model, optimizer = apex.amp.initialize(model, optimizer, opt_level = args.fp16, keep_batchnorm_fp32 = args.fp16_keep_batchnorm_fp32)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma = args.decay_gamma, milestones = args.decay_milestones) if args.scheduler == 'MultiStepLR' else optimizers.PolynomialDecayLR(optimizer, power = args.decay_power, decay_steps = len(train_data_loader) * args.decay_epochs, end_lr = args.decay_lr) if args.scheduler == 'PolynomialDecayLR' else torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.decay_step_size, gamma = args.decay_gamma) if args.scheduler == 'StepLR' else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: args.lr)
	if args.checkpoint:
		models.load_checkpoint(args.checkpoint, model, optimizer, train_sampler)
	os.makedirs(args.experiment_dir, exist_ok = True)
	tensorboard = torch.utils.tensorboard.SummaryWriter(os.path.join(args.experiment_dir, 'tensorboard'))
	with open(os.path.join(args.experiment_dir, args.args), 'w') as f:
		json.dump(vars(args), f, sort_keys = True, ensure_ascii = False, indent = 2)

	tic = time.time()
	iteration = 0
	loss_avg, time_ms_avg, lr_avg = 0.0, 0.0, 0.0
	moving_avg = lambda avg, x, max = 0, K = 50: (1. / K) * min(x, max) + (1 - 1./K) * avg
	for epoch in range(args.epochs if args.train_data_path else 0):
		model.train()
		for batch_idx, (inputs, targets, filenames, input_percentages, target_lengths) in enumerate(train_data_loader):
			toc = time.time()
			input_lengths = (input_percentages.cpu() * inputs.shape[-1]).int()
			logits = model(inputs.to(args.device), input_lengths)
			output_lengths = models.compute_output_lengths(model, input_lengths)
			loss = criterion(F.log_softmax(logits.permute(2, 0, 1), dim = -1), targets.to(args.device), output_lengths.to(args.device), target_lengths.to(args.device)) / len(inputs)
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
			print(f'{args.id} | epoch: {epoch:02d} iter: [{batch_idx: >6d} / {len(train_data_loader)} {iteration: >9d}] loss: {float(loss): 7.2f} <{loss_avg: 7.2f}> time: {time_ms_model:6.0f} <{time_ms_avg:4.0f}> +{time_ms_data:.2f} ms  | lr: {lr}')#<{lr:.0e}>')
			tic = time.time()
			scheduler.step()
			iteration += 1

			if args.val_iteration_interval is not None and iteration > 0 and iteration % args.val_iteration_interval == 0:
				evaluate_model(epoch, batch_idx, iteration)

			if iteration % args.log_iteration_interval == 0:
				tensorboard.add_scalars(args.id, {train_dataset_name + '_loss_avg' : loss_avg, train_dataset_name + '_lr_avg_x10K' : lr_avg * 1e4}, iteration)

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
	parser.add_argument('--epochs', type = int, default = 20)
	parser.add_argument('--num-input-features', default = 64)
	parser.add_argument('--train-data-path')
	parser.add_argument('--val-data-path', nargs = '+')
	parser.add_argument('--sample-rate', type = int, default = 16000)
	parser.add_argument('--window-size', type = float, default = 0.02)
	parser.add_argument('--window-stride', type = float, default = 0.01)
	parser.add_argument('--window', default = 'hann', choices = ['hann', 'hamming'])
	parser.add_argument('--num-workers', type = int, default = 10)
	parser.add_argument('--train-batch-size', type = int, default = 64)
	parser.add_argument('--val-batch-size', type = int, default = 64)
	parser.add_argument('--device', default = 'cuda', choices = ['cuda', 'cpu'])
	parser.add_argument('--checkpoint')
	parser.add_argument('--experiments-dir', default = 'data/experiments')
	parser.add_argument('--experiment-dir', default = '{experiments_dir}/{id}')
	parser.add_argument('--transcripts', default = 'data/transcripts_{val_dataset_name}.json')
	parser.add_argument('--logits', default = 'data/logits_{val_dataset_name}.pt')
	parser.add_argument('--args', default = 'args.json')
	parser.add_argument('--model', default = 'Wav2LetterRu')
	parser.add_argument('--seed', type = int, default = 1)
	parser.add_argument('--id', default = '{model}_{optimizer}_lr{lr:.0e}_wd{weight_decay:.0e}_bs{train_batch_size}_{name}')
	parser.add_argument('--name', default = '')
	parser.add_argument('--dry', action = 'store_true')
	parser.add_argument('--lang', default = 'ru')
	parser.add_argument('--val-waveform-transforms', default = '')
	parser.add_argument('--train-waveform-transforms', default = '')
	parser.add_argument('--val-iteration-interval', type = int, default = None)
	parser.add_argument('--log-iteration-interval', type = int, default = 100)
	parser.add_argument('--augment', action = 'store_true')
	parser.add_argument('--verbose', action = 'store_true')
	traintest(parser.parse_args())
