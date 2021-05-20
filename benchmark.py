import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint')
parser.add_argument('--device', default = 'cuda', help = 'TODO: proper device choosing')
parser.add_argument('--iterations', type = int, default = 16)
parser.add_argument('--iterations-warmup', type = int, default = 16)
parser.add_argument('--frontend', action = 'store_true')
parser.add_argument('--fp16', choices = ['', 'O0', 'O1', 'O2', 'O3'], default = None)
parser.add_argument(
	'--num-input-features', default = 64, help = 'num of mel-scale features produced by logfbank frontend'
)
parser.add_argument('--sample-rate', type = int, default = 8_000, help = 'for frontend')
parser.add_argument('--window-size', type = float, default = 0.02, help = 'for frontend, in seconds')
parser.add_argument('--window-stride', type = float, default = 0.01, help = 'for frontend, in seconds')
parser.add_argument('--lang', default = 'ru')
parser.add_argument(
	'--window', default = 'hann_window', choices = ['hann_window', 'hamming_window'], help = 'for frontend'
)
parser.add_argument('--model', default = 'JasperNetBig')
parser.add_argument('--onnx')
parser.add_argument('--onnx-output-node-name', type = str, default = 'logits')
parser.add_argument('--stft-mode', choices = ['conv', ''], default = '')
parser.add_argument('-B', type = int, default = 256)
parser.add_argument('-T', type = float, default = 5.12)
parser.add_argument('--profile-cuda', action = 'store_true')
parser.add_argument('--save-cudnn-cublas-logs', action = 'store_true', help='Save CUDNN CUBLAS logs')
parser.add_argument('--cudnn-logdest', type=str, default = 'cudnn_log.txt', help='Destination path to CUDNN logs')
parser.add_argument('--cublas-logdest', type=str, default = 'cublas_log.txt', help='Destination path to CUBLAS logs')

parser.add_argument('--profile-pyprof', action = 'store_true')
parser.add_argument('--profile-autograd')
parser.add_argument('--data-parallel', action = 'store_true')
parser.add_argument('--backward', action = 'store_true')
parser.add_argument('--input-time-dim-multiple', type = int, default = 128)

parser.add_argument('--output-path', '-o')

args = parser.parse_args()

if args.save_cudnn_cublas_logs:
	os.environ["CUDNN_LOGINFO_DBG"] = '1'
	os.environ["CUBLAS_LOGINFO_DBG"] = '1'
	os.environ["CUDNN_LOGDEST_DBG"] = args.cudnn_logdest
	os.environ["CUBLAS_LOGDEST_DBG"] = args.cublas_logdest

import gc
import math
import time
import torch
import torch.cuda.profiler
import onnxruntime
import models
import datasets
import utils

checkpoint = torch.load(args.checkpoint, map_location = 'cpu') if args.checkpoint else None
if checkpoint:
	args.model, args.lang, args.sample_rate, args.window_size, args.window_stride, args.window, args.num_input_features = map(
		checkpoint['args'].get,
		['model', 'lang', 'sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features'])

use_cuda = 'cuda' in args.device


if args.onnx:
	onnxruntime_session = onnxruntime.InferenceSession(args.onnx)

	def load_batch_to_device(batch, device):
		batch = batch.numpy()
		io_binding = onnxruntime_session.io_binding()
		batch_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(batch, device, 0)
		io_binding.bind_input(
			name = 'x',
			device_type = batch_ortvalue.device_name(),
			device_id = 0,
			element_type = batch.dtype,
			shape = batch_ortvalue.shape(),
			buffer_ptr = batch_ortvalue.data_ptr()
		)
		io_binding.bind_output(args.onnx_output_node_name, device)
		return io_binding

	load_batch = lambda batch: load_batch_to_device(batch, args.device)
	model = lambda io_binding: onnxruntime_session.run_with_iobinding(io_binding)

else:
	frontend = models.LogFilterBankFrontend(
		args.num_input_features,
		args.sample_rate,
		args.window_size,
		args.window_stride,
		args.window,
		stft_mode = args.stft_mode,
	) if args.frontend else None
	model = getattr(models, args.model)(
		args.num_input_features, [len(labels)],
		frontend = frontend,
		dict = lambda logits,
		log_probs,
		olen,
		**kwargs: logits[0]
	)
	if checkpoint:
		model.load_state_dict(checkpoint['model_state_dict'], strict = False)
	model.to(args.device)

	if not args.backward:
		model.eval()
		model.fuse_conv_bn_eval()

	model, *_ = models.data_parallel_and_autocast(model, opt_level = args.fp16, data_parallel = args.data_parallel)
	load_batch = lambda x: x.to(args.device, non_blocking = True)

tictoc = lambda: (use_cuda and torch.cuda.synchronize()) or time.time()

batch_shape = [args.B, int(args.T * args.sample_rate)
				] if args.frontend else [args.B, args.num_input_features, int(args.T / args.window_stride)]

if args.input_time_dim_multiple and batch_shape[-1] % args.input_time_dim_multiple != 0:
	batch_shape[-1] = int(math.ceil(batch_shape[-1] / args.input_time_dim_multiple) * args.input_time_dim_multiple)

example_time = batch_shape[-1] / args.sample_rate if args.frontend else batch_shape[-1] * args.window_stride

batch = torch.rand(*batch_shape)

print(args)
print()
print(
	'batch {} | stride {:.02f} sec | audio {:.02f} sec'.format(
		'x'.join(map(str, batch_shape)), args.window_stride, args.B * example_time
	)
)
print()

print('Warming up for', args.iterations_warmup, 'iterations')
tic_wall = tictoc()
if use_cuda:
	torch.backends.cudnn.benchmark = True

torch.set_grad_enabled(args.backward)
loaded_batch = load_batch(batch)
for i in range(args.iterations_warmup):
	y = model(loaded_batch)
	if args.backward:
		y.sum().backward()

print('Warmup done in {:.02f} wall clock seconds'.format(tictoc() - tic_wall))
print()

if args.profile_pyprof or args.profile_cuda:
	import pyprof
	pyprof.nvtx.nvmarker.patch_apex_module = lambda modstr, old = pyprof.nvtx.nvmarker.patch_apex_module: old(
			modstr) if modstr != 'apex.contrib.multihead_attn' else None  # coused by this bug https://github.com/NVIDIA/apex/issues/958 https://github.com/NVIDIA/PyProf/issues/167
	pyprof.nvtx.init()
if args.profile_cuda:
	torch.autograd.profiler.emit_nvtx()
	torch.cuda.profiler.start()
if args.profile_autograd:
	autograd_profiler = torch.autograd.profiler.profile(
		use_cuda = use_cuda, enabled = True, profile_memory = True, record_shapes = True
	)
	autograd_profiler.__enter__()

print('Starting benchmark for', args.iterations, 'iterations:', 'fwd', '+ bwd' if args.backward else '')
tic_wall = tictoc()
times_fwd, times_bwd, fragmentation, global_device_memory_utilisation = \
	torch.zeros(args.iterations), torch.zeros(args.iterations), \
		torch.zeros(args.iterations), torch.zeros(args.iterations)

loaded_batch = load_batch(batch)
for i in range(args.iterations):
	tic = tictoc()
	y = model(loaded_batch)
	toc = tictoc()
	global_device_memory_utilisation[i] = utils.compute_global_device_memory_utilisation()
	fragmentation[i] = utils.compute_memory_fragmentation()
	if args.backward:
		y.sum().backward()
	tac = tictoc()
	times_fwd[i] = toc - tic
	times_bwd[i] = tac - toc
	y = None

mem_reserved = torch.cuda.max_memory_reserved(args.device) if use_cuda else 0
mem_allocated = torch.cuda.max_memory_allocated(args.device) if use_cuda else 0
print('Benchmark done in {:.02f} wall clock seconds'.format(tictoc() - tic_wall))
print()

if args.profile_autograd:
	autograd_profiler.__exit__(None, None, None)
	autograd_profiler.export_chrome_trace(args.profile_autograd)
print(f'B: {args.B}, T: {args.T}')
print(
	'load+fwd {:.02f} msec | bwd {:.02f} msec | cudamemreserved {:.02f} mb | cudamemallocated {:.02f} mb | '
	'cudamemutilization: {:.02f} | global_dev_memutilization: {:.02f} | rtf: {:.02f}'
	.format(
		float(times_fwd.mean()) * 1e3,
		float(times_bwd.mean()) * 1e3,
		mem_reserved * 1e-6,
		mem_allocated * 1e-6,
		float(fragmentation.mean()),
		float(global_device_memory_utilisation.mean()),
		float(args.B * example_time * args.iterations / times_fwd.sum())
	)
)

if args.output:
	with open(args.output, 'a+') as f:
		message = f'{args.B}\t' \
		           f'{args.T}\t' \
		           f'{times_fwd.mean()}\t' \
		           f'{times_fwd.sum()}\t' \
		           f'{fragmentation.mean()}\t' \
		           f'{args.B * example_time * args.iterations / 3600.0}\t' \
		           f'{args.B * example_time * args.iterations / times_fwd.sum()}\t' \
		           f'{args.checkpoint}\t' \
		           f'{args.onnx}\t' \
		           f'\n'
		f.write(message)