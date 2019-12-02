import argparse
import importlib
import time
import torch
import models
import dataset
import apex

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint')
parser.add_argument('--device', default = 'cuda')
parser.add_argument('--iterations', type = int, default = 128)
parser.add_argument('--iterations-warmup', type = int, default = 16)
parser.add_argument('--frontend', action = 'store_true')
parser.add_argument('--fp16', choices = ['O0', 'O1', 'O2', 'O3'], default = None)
parser.add_argument('--num-input-features', default = 64, help = 'num of mel-scale features produced by logfbank frontend')
parser.add_argument('--sample-rate', type = int, default = 8_000, help = 'for frontend')
parser.add_argument('--window-size', type = float, default = 0.02, help = 'for frontend, in seconds')
parser.add_argument('--window-stride', type = float, default = 0.01, help = 'for frontend, in seconds')
parser.add_argument('--lang', default = 'ru')
parser.add_argument('--window', default = 'hann_window', choices = ['hann_window', 'hamming_window'], help = 'for frontend')
parser.add_argument('--model', default = 'JasperNetBigInplace')
parser.add_argument('-B', type = int, default = 128)
parser.add_argument('-T', type = int, default = 10)
args = parser.parse_args()

torch.set_grad_enabled(False)

checkpoint = torch.load(args.checkpoint, map_location = 'cpu') if args.checkpoint else None
if checkpoint:
	args.model, args.lang, args.sample_rate, args.window_size, args.window_stride, args.window, args.num_input_features = map(checkpoint['args'].get, ['model', 'lang', 'sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features'])

labels = dataset.Labels(importlib.import_module(args.lang))
frontend = models.LogFilterBank(args.num_input_features, args.sample_rate, args.window_size, args.window_stride, args.window)
model = getattr(models, args.model)(args.num_input_features, [len(labels)], frontend = frontend if args.frontend else None)
if checkpoint:
	model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(args.device)
model.eval()
model.fuse_conv_bn_eval()
if args.fp16:
	model = apex.amp.initialize(model, opt_level = args.fp16)

tictoc = lambda: (torch.cuda.is_available and torch.cuda.synchronize()) or time.time()

batch = torch.rand(args.B, args.T * args.sample_rate, device = args.device) if args.frontend else torch.rand(args.B, args.num_input_features, int(args.T / args.window_stride), device = args.device) 

for i in range(args.iterations_warmup):
	model(batch)

times = torch.FloatTensor(args.iterations)
for i in range(args.iterations):
	tic = tictoc()
	model(batch)
	times[i] = tictoc() - tic

print(args)
print('batch {} | stride {:.02f} sec | audio {:.02f} sec | mean time {:.02f} msec'.format('x'.join(map(str, batch.shape)), args.window_stride, args.B * args.T, float(times.mean()) * 1000))
