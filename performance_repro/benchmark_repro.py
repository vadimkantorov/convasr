import math
import argparse
import time
import torch
import torch.cuda.profiler
import apex
import onnxruntime
import models


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
parser.add_argument('--checkpoint')
parser.add_argument('--device', default='cuda', help='TODO: proper device choosing')
parser.add_argument('--iterations', type=int, default=16)
parser.add_argument('--iterations-warmup', type=int, default=16)
parser.add_argument('--frontend', action='store_true')
parser.add_argument('--fp16', choices=['', 'O0', 'O1', 'O2', 'O3'], default=None)
parser.add_argument('--num-input-features', default=64, help='num of mel-scale features produced by logfbank frontend')
parser.add_argument('--sample-rate', type=int, default=8_000, help='for frontend')
parser.add_argument('--window-size', type=float, default=0.02, help='for frontend, in seconds')
parser.add_argument('--window-stride', type=float, default=0.01, help='for frontend, in seconds')
parser.add_argument('--lang', default='ru')
parser.add_argument('--window', default='hann_window', choices=['hann_window', 'hamming_window'], help='for frontend')
parser.add_argument('--model', default='OneConvModel')
parser.add_argument('--onnx')
parser.add_argument('--stft-mode', choices=['conv', ''], default='')
parser.add_argument('-B', type=int, default=256)
parser.add_argument('-T', type=int, default=5.12)
parser.add_argument('--profile-cuda', action='store_true')
parser.add_argument('--profile-autograd')
parser.add_argument('--data-parallel', action='store_true')
parser.add_argument('--backward', action='store_true')
parser.add_argument('--set-cudnn-log', action='store_true')
args = parser.parse_args()


dtype = torch.float16
use_cuda = 'cuda' in args.device


if args.onnx:
	export_onnx(args.onnx)

	onnxruntime_session = onnxruntime.InferenceSession(args.onnx)
	model = lambda x: onnxruntime_session.run(None, dict(input=x))
	load_batch = lambda x: x.numpy()

else:
	model = get_model()
	model.to(args.device)

	if not args.backward:
		model.eval()

	model, *_ = models.data_parallel_and_autocast(model, opt_level=args.fp16, data_parallel=args.data_parallel)
	load_batch = lambda x: x.to(args.device, non_blocking=True)

tictoc = lambda: (use_cuda and torch.cuda.synchronize()) or time.time()

batch_shape = [args.B, args.num_input_features, int(args.T / args.window_stride)]

if batch_shape[-1] % 128 != 0:
	batch_shape[-1] = int(math.ceil(batch_shape[-1] / 128) * 128)

example_time = batch_shape[-1] * args.window_stride

batch = torch.rand(*batch_shape, dtype=torch.float16)
batch = batch.pin_memory()

print(args)


print('Warming up for', args.iterations_warmup, 'iterations')
tic_wall = tictoc()
torch.backends.cudnn.benchmark = True

torch.set_grad_enabled(args.backward)
for i in range(args.iterations_warmup):
	y = model(load_batch(batch))
	if args.backward:
		y.sum().backward()

print('Warmup done in {:.02f} wall clock seconds'.format(tictoc() - tic_wall))
print()


if args.profile_cuda:
	apex.pyprof.nvtx.init()
	torch.autograd.profiler.emit_nvtx()
	torch.cuda.profiler.start()
if args.profile_autograd:
	autograd_profiler = torch.autograd.profiler.profile(use_cuda=use_cuda, enabled=True, profile_memory=True, 			record_shapes=True)
	autograd_profiler.__enter__()

print('Starting benchmark for', args.iterations, 'iterations:', 'fwd', '+ bwd' if args.backward else '')
times_fwd = torch.zeros(args.iterations)
for i in range(args.iterations):
	batch = torch.rand(*batch_shape, dtype=torch.float16)
	batch = batch.pin_memory()

	tic = tictoc()
	y = model(load_batch(batch))
	toc = tictoc()
	if args.backward:
		y.sum().backward()
	tac = tictoc()
	times_fwd[i] = toc - tic
	y = None

if args.profile_autograd:
	autograd_profiler.__exit__(None, None, None)
	autograd_profiler.export_chrome_trace(args.profile_autograd)

print(f'B: {args.B}, T: {args.T}')
print(
		'load+fwd {:.02f} msec | rtf: {:.02f}'
			.format(
				float(times_fwd.mean()) * 1e3,
				float(args.B * example_time * args.iterations / times_fwd.sum())
		)
)
