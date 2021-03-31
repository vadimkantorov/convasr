import math
import argparse
import time
import torch
import torch.cuda.profiler
import apex
import onnxruntime


def get_model():
	return torch.nn.Conv1d(
			in_channels=64,
			out_channels=256,
			kernel_size=11,
			stride=2,
			padding=(1 * 11 // 2),
			dilation=1,
			groups=1
	)


def export_onnx(onnx_path):
	model = get_model()

	torch.set_grad_enabled(False)
	model.eval()
	model.to('cuda')

	model = model.to(dtype)
	waveform_input = torch.rand((4, 64, 128), device='cuda', dtype=dtype)

	logits = model(waveform_input)

	torch.onnx.export(
			model, (waveform_input),
			onnx_path,
			opset_version=12,
			export_params=True,
			do_constant_folding=True,
			input_names=['input'],
			dynamic_axes=dict(input={
				0: 'B',
				2: 'T'
			})
	)

	## check correctness
	onnxruntime_session = onnxruntime.InferenceSession(onnx_path)
	(logits_,) = onnxruntime_session.run(None, dict(input=waveform_input.cpu().to(dtype=dtype).numpy()))

	assert torch.allclose(
			logits.cpu(),
			torch.from_numpy(logits_),
			**{
				'rtol': 1e-01,
				'atol': 1e-02
			}
	)


parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=16)
parser.add_argument('--iterations-warmup', type=int, default=16)
parser.add_argument('--fp16', default='O2')
parser.add_argument('--num-input-features', default=64)
parser.add_argument('--sample-rate', type=int, default=8_000, help='for frontend')
parser.add_argument('--window-stride', type=float, default=0.01, help='for frontend, in seconds')
parser.add_argument('--onnx')
parser.add_argument('-B', type=int)
parser.add_argument('-T', type=int)
parser.add_argument('--profile-cuda', action='store_true')
parser.add_argument('--profile-autograd')
args = parser.parse_args()

print(args)

dtype = torch.float16
use_cuda = True

if args.onnx:
	export_onnx(args.onnx)
	onnxruntime_session = onnxruntime.InferenceSession(args.onnx)
	model = lambda x: onnxruntime_session.run(None, dict(input=x))
	load_batch = lambda x: x.numpy()
else:
	model = get_model()
	model.to('cuda')
	model.eval()
	model = apex.amp.initialize(model, opt_level = args.fp16)
	#model, *_ = models.data_parallel_and_autocast(model, opt_level=args.fp16, data_parallel=args.data_parallel)
	load_batch = lambda x: x.to('cuda', non_blocking=True)

tictoc = lambda: (use_cuda and torch.cuda.synchronize()) or time.time()

batch_shape = [args.B, args.num_input_features, int(args.T / args.window_stride)]

if batch_shape[-1] % 128 != 0:
	batch_shape[-1] = int(math.ceil(batch_shape[-1] / 128) * 128)

example_time = batch_shape[-1] * args.window_stride

batch = torch.rand(*batch_shape, dtype=torch.float16)
batch = batch.pin_memory()

print('Warming up for', args.iterations_warmup, 'iterations')
tic_wall = tictoc()
torch.backends.cudnn.benchmark = True

for i in range(args.iterations_warmup):
	y = model(load_batch(batch))

print('Warmup done in {:.02f} wall clock seconds'.format(tictoc() - tic_wall))
print()

if args.profile_cuda:
	apex.pyprof.nvtx.init()
	torch.autograd.profiler.emit_nvtx()
	torch.cuda.profiler.start()
if args.profile_autograd:
	autograd_profiler = torch.autograd.profiler.profile(use_cuda=use_cuda, enabled=True, profile_memory=True,
			record_shapes=True)
	autograd_profiler.__enter__()

print('Starting benchmark for', args.iterations, 'iterations:', 'fwd')
times_fwd = torch.zeros(args.iterations)
batch = torch.rand(*batch_shape, dtype=torch.float16)
batch = batch.pin_memory()

for i in range(args.iterations):
	tic = tictoc()
	y = model(load_batch(batch))
	times_fwd[i] = tictoc() - tic
	y = None

if args.profile_autograd:
	autograd_profiler.__exit__(None, None, None)
	autograd_profiler.export_chrome_trace(args.profile_autograd)

print('Batch shape', batch_shape)
print('load+fwd {:.02f} msec | rtf: {:.02f}'
	.format(
		float(times_fwd.mean()) * 1e3,
		float(args.B * example_time * args.iterations / times_fwd.sum())
)
)
