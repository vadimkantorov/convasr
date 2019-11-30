import argparse
import importlib
import time
import torch
import models
import dataset
import apex

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required = True)
parser.add_argument('--device', default = 'cuda')
parser.add_argument('--iterations', type = int, default = 16)
parser.add_argument('--fp16', choices = ['O0', 'O1', 'O2', 'O3'], default = None)
parser.add_argument('-B', type = int, default = 512)
parser.add_argument('-T', type = int, default = 512)
args = parser.parse_args()

torch.set_grad_enabled(False)

checkpoint = torch.load(args.checkpoint, map_location = 'cpu')
window_stride, num_input_features = map(checkpoint['args'].get, ['window_stride', 'num_input_features'])
labels = dataset.Labels(importlib.import_module(checkpoint['lang']))
model = getattr(models, checkpoint['model'])(num_input_features, [len(labels)])
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(args.device)
model.eval()
model.fuse_conv_bn_eval()

if args.fp16:
	model = apex.amp.initialize(model, opt_level = args.fp16)

tictoc = lambda: (torch.cuda.synchronize() if torch.cuda.is_available else False) or time.time()

batch = torch.rand(args.B, num_input_features, args.T, device = args.device)
for i in range(args.iterations):
	model(batch)

times = torch.FloatTensor(args.iterations)
for i in range(args.iterations):
	tic = tictoc()
	model(batch)
	times[i] = tictoc() - tic

print(args)
print('batch {} | stride {:.02f} sec | audio {:.02f} sec | mean time {:.02f} msec'.format('x'.join(map(str, batch.shape)), window_stride, (args.B * args.T) * window_stride, float(times.mean()) * 1000))
