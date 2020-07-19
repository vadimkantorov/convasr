import gc
import math
import argparse
import time
import torch
import torch.cuda.profiler
import apex
import onnxruntime
import models
import datasets

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint')
parser.add_argument('--device', default = 'cuda', help ='TODO: proper device choosing')
parser.add_argument('--iterations', type = int, default = 32)
parser.add_argument('--iterations-warmup', type = int, default = 16)
parser.add_argument('--frontend', action = 'store_true')
parser.add_argument('--fp16', choices = ['', 'O0', 'O1', 'O2', 'O3'], default = None)
parser.add_argument('--num-input-features', default = 64, help = 'num of mel-scale features produced by logfbank frontend')
parser.add_argument('--sample-rate', type = int, default = 8_000, help = 'for frontend')
parser.add_argument('--window-size', type = float, default = 0.02, help = 'for frontend, in seconds')
parser.add_argument('--window-stride', type = float, default = 0.01, help = 'for frontend, in seconds')
parser.add_argument('--lang', default = 'ru')
parser.add_argument('--window', default = 'hann_window', choices = ['hann_window', 'hamming_window'], help = 'for frontend')
parser.add_argument('--model', default = 'JasperNetBig')
parser.add_argument('--onnx')
parser.add_argument('--stft-mode', choices = ['conv', ''], default = '')
parser.add_argument('-B', type = int, default = 128)
parser.add_argument('-T', type = int, default = 5.12)
parser.add_argument('--profile-cuda', action = 'store_true')
parser.add_argument('--profile-autograd')
parser.add_argument('--data-parallel', action = 'store_true')
parser.add_argument('--backward', action = 'store_true')
parser.add_argument('--no-grad', action = 'store_true')
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint, map_location = 'cpu') if args.checkpoint else None
if checkpoint:
	args.model, args.lang, args.sample_rate, args.window_size, args.window_stride, args.window, args.num_input_features = map(checkpoint['args'].get, ['model', 'lang', 'sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features'])

use_cuda = 'cuda' in args.device

labels = datasets.Labels(datasets.Language(args.lang))

if args.no_grad:
	torch.set_grad_enabled(False)

if args.onnx:
	onnxruntime_session = onnxruntime.InferenceSession(args.onnx)
	model = lambda x: onnxruntime_session.run(None, dict(x = x))
	load_batch = lambda x: x.numpy()

else:
	frontend = models.LogFilterBankFrontend(args.num_input_features, args.sample_rate, args.window_size, args.window_stride, args.window, stft_mode = args.stft_mode) if args.frontend else None
	model = getattr(models, args.model)(args.num_input_features, [len(labels)], frontend = frontend, dict = lambda logits, log_probs, olen, **kwargs: logits[0])
	if checkpoint:
		model.load_state_dict(checkpoint['model_state_dict'])
	model.to(args.device)
	
	if args.no_grad:
		model.eval()
		model.fuse_conv_bn_eval()
	
	model, *_ = models.data_parallel_and_autocast(model, opt_level = args.fp16) if args.data_parallel else (model,)
	load_batch = lambda x: x.to(args.device, non_blocking = True)

tictoc = lambda: (use_cuda and torch.cuda.synchronize()) or time.time()

batch_shape = [args.B, args.T * args.sample_rate] if args.frontend else [args.B, args.num_input_features, int(args.T / args.window_stride)]
if batch_shape[-1] % 128 != 0:
	batch_shape[-1] = int(math.ceil(batch_shape[-1] / 128) * 128)
example_time = batch_shape[-1] / args.sample_rate if args.frontend else batch_shape[-1] * args.window_stride

batch = torch.rand(*batch_shape)
batch = batch.pin_memory()
	
print(args)
print()
print('batch {} | stride {:.02f} sec | audio {:.02f} sec'.format('x'.join(map(str, batch_shape)), args.window_stride, args.B * example_time))
print()

print('Warming up for', args.iterations_warmup, 'iterations')
tic_wall = tictoc()
for i in range(args.iterations_warmup):
	model(load_batch(batch))
print('Warmup done in {:.02f} wall clock seconds'.format(tictoc() - tic_wall))
print()

if args.profile_cuda:
	apex.pyprof.nvtx.init()
	torch.autograd.profiler.emit_nvtx()
	torch.cuda.profiler.start()
if args.profile_autograd:
	autograd_profiler = torch.autograd.profiler.profile(use_cuda = use_cuda, enabled = True)
	autograd_profiler.__enter__()

print('Starting benchmark for', args.iterations, 'iterations:', 'fwd', '+ bwd' if args.backward else '')
tic_wall = tictoc()
times_fwd, times_bwd, fragmentation = torch.zeros(args.iterations), torch.zeros(args.iterations), torch.zeros(args.iterations)
for i in range(args.iterations):
	tic = tictoc()
	y = model(load_batch(batch))
	toc = tictoc()
	fragmentation[i] = models.compute_memory_fragmentation()
	if args.backward:
		y.sum().backward()
	tac = tictoc()
	times_fwd[i] = toc - tic
	times_bwd[i] = tac - toc

	#for p in model.parameters():
	#	p.grad = None
	#gc.collect()

mem_reserved = torch.cuda.max_memory_reserved(args.device) if use_cuda else 0
mem_allocated = torch.cuda.max_memory_allocated(args.device) if use_cuda else 0
print('Benchmark done in {:.02f} wall clock seconds'.format(tictoc() - tic_wall))
print()

if args.profile_autograd:
	autograd_profiler.__exit__(None, None, None)
	autograd_profiler.export_chrome_trace(args.profile_autograd)

print('load+fwd {:.02f} msec | bwd {:.02f} msec | cudamemreserved {:.02f} mb | cudamemallocated {:.02f} mb | cudamemutilization: {:.02f}'.format(float(times_fwd.mean()) * 1e3, float(times_bwd.mean()) * 1e3, mem_reserved * 1e-6, mem_allocated * 1e-6, float(fragmentation.mean())))
