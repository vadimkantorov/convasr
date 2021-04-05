import argparse
import os
import math
import time
import json

import torch
import torch.cuda.profiler
import onnxruntime
import numpy as np

import models
import text_processing


def main(args):
	use_cuda = 'cuda' in args.device
	if use_cuda:
		print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES")!r}')
	print('initializing model...')

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

		text_config = json.load(open(args.text_config))
		text_pipelines = []
		for pipeline_name in args.text_pipelines:
			text_pipelines.append(text_processing.ProcessingPipeline.make(text_config, pipeline_name))

		frontend = models.LogFilterBankFrontend(
			args.num_input_features,
			args.sample_rate,
			args.window_size,
			args.window_stride,
			args.window,
			stft_mode=args.stft_mode
		)
		model = getattr(models, args.model)(
			num_input_features=args.num_input_features,
			num_classes=[pipeline.tokenizer.vocab_size for pipeline in text_pipelines],
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
	batch = torch.rand(args.B, batch_width)
	batch = batch.pin_memory()

	print(args, end='\n\n')
	print(f'batch [{args.B}, {batch_width}] | audio {args.B * example_time:.2f} sec\n')

	def tictoc():
		if use_cuda:
			torch.cuda.synchronize()
		return time.time()

	print(f'Warming up for {args.warmup_iterations} iterations...')
	tic_wall = tictoc()
	if use_cuda:
		torch.backends.cudnn.benchmark = True
	torch.set_grad_enabled(False)  # no back-prop - no gradients
	for i in range(args.warmup_iterations):
		model(load_batch(batch))
	print(f'Warmup done in {tictoc() - tic_wall:.1f} sec\n')

	n_requests = int(round(args.benchmark_duration * args.rps))
	print(f'Starting {args.benchmark_duration} second benchmark ({n_requests} requests, rps {args.rps:.1f})...')
	schedule = np.random.random(n_requests) * args.benchmark_duration  # uniform random distribution of requests
	schedule = np.sort(schedule) + tictoc()
	print(f'avg gap between requests: {np.diff(schedule).mean() * 1e3:.1f} ms')
	latency_times, idle_times = [], []
	slow_warning = False
	for t_request in schedule:
		tic = tictoc()
		if tic < t_request:  # no requests yet. at this point prod would wait for the next request
			sleep_time = t_request - tic
			idle_times.append(sleep_time)
			time.sleep(sleep_time)
		elif tic > t_request + 0.5 and not slow_warning:
			print(f'model is too slow and can\'t handle {args.rps} requests per second!')
			slow_warning = True
		model(load_batch(batch))
		latency_times.append(tictoc() - t_request)
	latency_times = np.array(latency_times) * 1e3  # convert to ms
	print(
		f'Latency mean: {latency_times.mean():.1f} ms, ' +
		f'median: {np.quantile(latency_times, .50):.1f} ms, ' +
		f'90-th percentile: {np.quantile(latency_times, .90):.1f} ms, ' +
		f'95-th percentile: {np.quantile(latency_times, .95):.1f} ms, ' +
		f'99-th percentile: {np.quantile(latency_times, .99):.1f} ms, ' +
		f'max: {latency_times.max():.1f} ms | ' +
		f'service idle time fraction: {sum(idle_times) / args.benchmark_duration:.1%}'
	)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--onnx')
	parser.add_argument('--checkpoint')
	parser.add_argument('--text-config', default='configs/ru_text_config.json')
	parser.add_argument('--text-pipelines', nargs='+', default=['char_legacy'],
		help='text processing pipelines (names should be defined in text-config)')
	parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
	parser.add_argument('--warmup-iterations', type=int, default=100)
	parser.add_argument('--benchmark-duration', type=int, default=30, help='benchmark duration in seconds')
	parser.add_argument('--rps', type=float, default=60, help='requests per second')
	parser.add_argument('--fp16', choices=['', 'O0', 'O1', 'O2', 'O3'], default='O2')
	parser.add_argument('--stft-mode', choices=['conv', ''], default='')
	parser.add_argument('-B', type=int, default=1)
	parser.add_argument('-T', type=float, default=6.0)
	main(parser.parse_args())
