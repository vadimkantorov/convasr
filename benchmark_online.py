import argparse
import os
import math
import time

import torch
import torch.cuda.profiler
import onnxruntime
import numpy as np

import models
import datasets
import utils


def main(args):
	use_cuda = 'cuda' in args.device

	if args.onnx:
		onnxruntime_session = onnxruntime.InferenceSession(args.onnx)
		model = lambda x: onnxruntime_session.run(None, dict(x=x))
		load_batch = lambda x: x.numpy()
		args.sample_rate = 8000
	else:
		assert args.checkpoint is not None and os.path.isfile(args.checkpoint)
		checkpoint = torch.load(args.checkpoint, map_location='cpu')
		args.model, args.lang, args.sample_rate, args.window_size, args.window_stride, args.window, args.num_input_features = (
			checkpoint['args'].get(key) for key in
			['model', 'lang', 'sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features']
		)
		labels = datasets.Labels(datasets.Language(args.lang))
		frontend = models.LogFilterBankFrontend(
			args.num_input_features,
			args.sample_rate,
			args.window_size,
			args.window_stride,
			args.window,
			stft_mode=args.stft_mode
		)
		model = getattr(models, args.model)(
			args.num_input_features, [len(labels)],
			frontend=frontend,
			dict=lambda logits, log_probs, olen, **kwargs: logits[0]
		)
		model.load_state_dict(checkpoint['model_state_dict'], strict=False)
		model.to(args.device)
		model.eval()
		model.fuse_conv_bn_eval()
		model, *_ = models.data_parallel_and_autocast(model, opt_level=args.fp16, data_parallel=False)
		load_batch = lambda x: x.to(args.device, non_blocking=True)

	batch_width = int(math.ceil(args.T * args.sample_rate / 128) * 128)
	example_time = batch_width / args.sample_rate
	batch_shape = [args.B, batch_width]
	batch = torch.rand(*batch_shape)
	batch = batch.pin_memory()

	print(args, end='\n\n')
	print(f'batch {batch_shape} | audio {args.B * example_time:.2f} sec\n')

	def tictoc():
		return torch.cuda.synchronize() if use_cuda else time.time()

	print(f'Warming up for {args.iterations_warmup} iterations')
	tic_wall = tictoc()
	if use_cuda:
		torch.backends.cudnn.benchmark = True
	torch.set_grad_enabled(False)  # no back-prop - no gradients
	for i in range(args.iterations_warmup):
		model(load_batch(batch))
	print(f'Warmup done in {tictoc() - tic_wall:.2f} sec\n')

	print(f'Starting benchmark for {args.iterations} iterations')
	tic_wall = tictoc()
	times_fwd, fragmentation = [], []
	for i in range(args.iterations):
		tic = tictoc()
		model(load_batch(batch))
		times_fwd.append(tictoc() - tic)
		fragmentation.append(utils.compute_memory_fragmentation())
	mem_reserved = torch.cuda.max_memory_reserved(args.device) if use_cuda else 0
	mem_allocated = torch.cuda.max_memory_allocated(args.device) if use_cuda else 0
	print(f'Benchmark done in {tictoc() - tic_wall:.2f} sec\n')

	print(
		f'load+fwd {np.mean(times_fwd) * 1e3:.2f} ms | ' +
		f'cudamemreserved {mem_reserved * 1e-6:.2f} MB | ' +
		f'cudamemallocated {mem_allocated * 1e-6:.2f} MB | ' +
		f'cudamemutilization: {np.mean(fragmentation):.2f} | ' +
		f'rtf: {args.B * example_time * args.iterations / sum(times_fwd):.2f}'
	)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--onnx')
	parser.add_argument('--checkpoint')
	parser.add_argument('--device', default='cuda', choices=['conv', 'cpu'])
	parser.add_argument('--warmup-iterations', type=int, default=16)
	parser.add_argument('--iterations', type=int, default=16)
	# parser.add_argument('--benchmark-duration', type=int, default=300, help='benchmark duration in seconds')
	parser.add_argument('--fp16', choices=['', 'O0', 'O1', 'O2', 'O3'], default='O2')
	parser.add_argument('--stft-mode', choices=['conv', ''], default='')
	parser.add_argument('-B', type=int, default=1)
	parser.add_argument('-T', type=int, default=6.0)
	main(parser.parse_args())
