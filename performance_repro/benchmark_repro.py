import argparse
import time
import torch
import torch.cuda.profiler
import onnxruntime
import numpy as np


def infer_ort(onnxruntime_session, io_binding):
	onnxruntime_session.run_with_iobinding(io_binding)
	ort_output_vals = io_binding.copy_outputs_to_cpu()[0]
	return ort_output_vals


def get_model(channels):
	return torch.nn.Conv1d(in_channels=channels, out_channels=channels*2, kernel_size=19,
			stride=2, padding=(1 * 11 // 2), dilation=1, groups=1)


def export_onnx(onnx_path, channels, dtype = torch.float16):
	model = get_model(channels)
	torch.set_grad_enabled(False)
	model.eval()
	model.to('cuda', dtype=dtype)
	waveform_input = torch.rand((4, channels, 128), device='cuda', dtype=dtype)

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

	## check export correctness
	onnxruntime_session = onnxruntime.InferenceSession(onnx_path)
	(logits_,) = onnxruntime_session.run(None, dict(input=waveform_input.cpu().to(dtype=dtype).numpy()))

	print((torch.from_numpy(logits_) - logits.cpu()).abs().to(dtype=torch.float32).max())
	assert torch.allclose(
			logits.cpu(),
			torch.from_numpy(logits_),
			**{
				'rtol': 1e-01,
				'atol': 1e-01
			}
	)

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=16)
parser.add_argument('--iterations-warmup', type=int, default=16)
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--onnx')
parser.add_argument('-B', type=int)
parser.add_argument('-T', type=int)
parser.add_argument('--profile-cuda', action='store_true')
parser.add_argument('--run-with-io-binding', action='store_true')
args = parser.parse_args()

print(args)

dtype = torch.float16
use_cuda = True

if args.onnx:
	export_onnx(args.onnx, args.channels, dtype)
	onnxruntime_session = onnxruntime.InferenceSession(args.onnx)
	model = lambda x: onnxruntime_session.run(None, dict(input=x))
	if args.run_with_io_binding:
		model = lambda io_binding: onnxruntime_session.run_with_iobinding(io_binding)
	pass
else:
	model = get_model(args.channels)
	model.to('cuda')
	model.eval()
	model.to(dtype=dtype)

tictoc = lambda: (use_cuda and torch.cuda.synchronize()) or time.time()

batch_shape = [args.B, args.channels, args.T]

batch = torch.rand(*batch_shape, dtype=torch.float16)

if args.onnx:
	io_binding = onnxruntime_session.io_binding()
	X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(batch.numpy(), 'cuda', 0)
	io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float16,
			shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
	io_binding.bind_output('3', 'cuda')
	batch = io_binding
	model = onnxruntime_session
	infer = lambda model, batch: infer_ort(model, batch)
else:
	batch = batch.to(device='cuda')
	infer = lambda model, batch: model(batch).cpu()

print('Warming up for', args.iterations_warmup, 'iterations')
tic_wall = tictoc()
torch.backends.cudnn.benchmark = True
for i in range(args.iterations_warmup):
	y = infer(model, batch)
print('Warmup done in {:.02f} wall clock seconds'.format(tictoc() - tic_wall))
print()

if args.profile_cuda:
	torch.cuda.profiler.start()

print('Starting benchmark for', args.iterations, 'iterations:', 'fwd')
times_fwd = torch.zeros(args.iterations)
batch = torch.rand(*batch_shape, dtype=torch.float16)

if args.onnx:
	io_binding = onnxruntime_session.io_binding()
	X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(batch.numpy(), 'cuda', 0)
	io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float16,
			shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
	io_binding.bind_output('3', 'cuda')
	batch = io_binding
	model = onnxruntime_session
	infer = lambda model, batch: infer_ort(model, batch)
else:
	batch = batch.to(device='cuda')
	infer = lambda model, batch: model(batch).cpu()

for i in range(args.iterations):
	tic = tictoc()
	y = infer(model, batch)
	times_fwd[i] = tictoc() - tic
	y = None

print('Batch shape', batch_shape)
print('average load+fwd {:.02f} msec'.format(float(times_fwd.mean()) * 1e3))

