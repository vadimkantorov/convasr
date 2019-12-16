import argparse
import importlib
import time
import torch
import models
import dataset
import apex
import torch.cuda.profiler
import onnxruntime

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint')
parser.add_argument('--devices', default = ['cuda'], nargs = '+')
parser.add_argument('--iterations', type = int, default = 128)
parser.add_argument('--iterations-warmup', type = int, default = 16)
parser.add_argument('--frontend', action = 'store_true')
parser.add_argument('--fp16', choices = ['', 'O0', 'O1', 'O2', 'O3'], default = None)
parser.add_argument('--num-input-features', default = 64, help = 'num of mel-scale features produced by logfbank frontend')
parser.add_argument('--sample-rate', type = int, default = 8_000, help = 'for frontend')
parser.add_argument('--window-size', type = float, default = 0.02, help = 'for frontend, in seconds')
parser.add_argument('--window-stride', type = float, default = 0.01, help = 'for frontend, in seconds')
parser.add_argument('--lang', default = 'ru')
parser.add_argument('--window', default = 'hann_window', choices = ['hann_window', 'hamming_window'], help = 'for frontend')
parser.add_argument('--model', default = 'JasperNetBigInplace')
parser.add_argument('--onnx')
parser.add_argument('--stft-mode', choices = ['conv', ''], default = '')
parser.add_argument('-B', type = int, default = 128)
parser.add_argument('-T', type = int, default = 10)
parser.add_argument('--profile-cuda', action = 'store_true')
parser.add_argument('--profile-autograd')
parser.add_argument('--dataparallel', action = 'store_true')
args = parser.parse_args()
args.devices = [0, 1]

torch.set_grad_enabled(False)

checkpoint = torch.load(args.checkpoint, map_location = 'cpu') if args.checkpoint else None
if checkpoint:
	args.model, args.lang, args.sample_rate, args.window_size, args.window_stride, args.window, args.num_input_features = map(checkpoint['args'].get, ['model', 'lang', 'sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features'])

labels = dataset.Labels(importlib.import_module(args.lang))

if args.onnx:
	onnxrt_session = onnxruntime.InferenceSession(args.onnx)
	model = lambda x: onnxrt_session.run(None, dict(x = x))
	load_batch = lambda x: x.numpy()
else:
	frontend = models.LogFilterBankFrontend(args.num_input_features, args.sample_rate, args.window_size, args.window_stride, args.window, stft_mode = args.stft_mode) if args.frontend else None
	model = getattr(models, args.model)(args.num_input_features, [len(labels)], frontend = frontend)
	if checkpoint:
		model.load_state_dict(checkpoint['model_state_dict'])
	model.to(args.devices[0])
	model.eval()
	model.fuse_conv_bn_eval()
	if args.fp16:
		model = apex.amp.initialize(torch.nn.Sequential(model), opt_level = args.fp16, enabled = bool(args.fp16))[0]
	if args.dataparallel:
		model = torch.nn.DataParallel(model, device_ids = args.devices)
	if args.fp16:
		model.forward = lambda *args, old_fwd = model.forward, input_caster = lambda tensor: tensor.to(apex.amp._amp_state.opt_properties.options['cast_model_type']), output_caster = lambda tensor: tensor.to(apex.amp._amp_state.opt_properties.options['cast_model_outputs'] if apex.amp._amp_state.opt_properties.options.get('cast_model_outputs') is not None else torch.float32), **kwargs: apex.amp._initialize.applier(old_fwd(*apex.amp._initialize.applier(args, input_caster), **apex.amp._initialize.applier(kwargs, input_caster)), output_caster)
	load_batch = lambda x: x.to(args.devices[0], non_blocking = True)

tictoc = lambda: (torch.cuda.is_available and torch.cuda.synchronize()) or time.time()

batch = torch.rand(args.B, args.T * args.sample_rate) if args.frontend else torch.rand(args.B, args.num_input_features, int(args.T / args.window_stride)) 
batch = batch.pin_memory()

for i in range(args.iterations_warmup):
	model(load_batch(batch))

if args.profile_cuda:
	apex.pyprof.nvtx.init()
	torch.autograd.profiler.emit_nvtx()
	torch.cuda.profiler.start()
if args.profile_autograd:
	autograd_profiler = torch.autograd.profiler.profile()
	autograd_profiler.__enter__()

times = torch.FloatTensor(args.iterations)
for i in range(args.iterations):
	tic = tictoc()
	model(load_batch(batch))
	times[i] = tictoc() - tic

if args.profile_autograd:
	autograd_profiler.__exit__(None, None, None)
	autograd_profiler.export_chrome_trace(args.profile_autograd)

print(args)
print('batch {} | stride {:.02f} sec | audio {:.02f} sec | load+proc {:.02f} msec'.format('x'.join(map(str, batch.shape)), args.window_stride, args.B * args.T, float(times.mean()) * 1000))
