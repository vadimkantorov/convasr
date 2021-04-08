""" This script is used to benchmark online-transcription speed and efficiency.
In online regime audio samples are shorter on average, but requests are unpredictable,
and two requests can occur at approximately the same time. To simulate this, uniformly random requests schedule
is generated in advance, and each response latency is recorded.

EXAMPLE:
export CUDA_VISIBLE_DEVICES=1
python benchmark_online.py \
--checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs512___\
finetune_after_self_train_on_2020_11_27_clean_valset_epoch144_iter0335000.pt \
--device cuda -T 6.0 --benchmark-duration 60 --rps 5.0

OUT:
initializing model...
batch [1, 48000] | audio 6.00 sec
Warming up for 100 iterations...
Warmup done in 2.7 sec
Starting 60 second benchmark (300 requests, rps 5.0)...
avg gap between requests: 195.0 ms
Latency mean: 33.6 ms, median: 33.1 ms, 90-th percentile: 36.9 ms, 95-th percentile: 46.2 ms,
99-th percentile: 60.7 ms, max: 73.0 ms | service idle time fraction: 81.7%

ONNX EXAMPLE:
export CUDA_VISIBLE_DEVICES=1
python benchmark_online.py \
--onnx best_checkpoints/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs512____\
rerun_finetune_after_self_train_epoch183_iter0290000.pt.12.fp16_masking_1.7.1.onnx \
--device cuda -T 6.0 --benchmark-duration 60 --rps 5.0

OUT:
initializing model...
initial providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
changed providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
batch [1, 48000] | audio 6.00 sec
Warming up for 100 iterations...
Warmup done in 2.5 sec
Starting 60 second benchmark (300 requests, rps 5.0)...
avg gap between requests: 199.1 ms
Latency mean: 22.7 ms, median: 21.7 ms, 90-th percentile: 23.8 ms, 95-th percentile: 32.3 ms,
99-th percentile: 42.5 ms, max: 61.2 ms | service idle time fraction: 88.9%

Tests show, that current model on 1 GPU can handle RPS=50 (x100 of our production), with peak latency < 500ms.
"""

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
		# todo: pass dict with provider setting when we will migrate to onnxruntime>=1.7
		onnxruntime_session = onnxruntime.InferenceSession(args.onnx)
		if args.device == 'cpu':
			onnxruntime_session.set_providers(['CPUExecutionProvider'])
		model = lambda x: onnxruntime_session.run(None, dict(x=x, xlen=[1.0] * len(x)))
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
		if use_cuda:
			model, *_ = models.data_parallel_and_autocast(model, opt_level=args.fp16, data_parallel=False)
		load_batch = lambda x: x.to(args.device, non_blocking=True)

	batch_width = int(math.ceil(args.T * args.sample_rate / 128) * 128)
	example_time = batch_width / args.sample_rate
	batch = torch.rand(args.B, batch_width)
	batch = batch.pin_memory()

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
		elif tic > t_request + 1.0 and not slow_warning:
			print(f'model is too slow and can\'t handle {args.rps} requests per second!')
			slow_warning = True
		logits = model(load_batch(batch))
		if not args.onnx:
			logits.cpu()
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
