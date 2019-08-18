import os
import collections
import glob
import json
import argparse
import base64
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def tra(transcripts):
	ref_tra = list(sorted(json.load(open(transcripts)), key = lambda j: j['cer']))
	vis = open(transcripts + '.html' , 'w')
	vis.write(f'<html><meta charset="utf-8"><body><h1>{args.transcripts}</h1><table style="border-collapse:collapse"><thead><tr><th>cer</th><th>filename</th><th>audio</th><th><div>reference</div><div>transcript</div></th></tr></thead><tbody>')

	for i, (reference, transcript, filename, cer) in enumerate(list(map(j.get, ['reference', 'transcript', 'filename', 'cer'])) for j in ref_tra):
		encoded = base64.b64encode(open(filename, 'rb').read()).decode('utf-8').replace('\n', '')
		vis.write(f'<tr><td style="border-right: 2px black solexperiment_id">{cer:.02%}</td> <td style="font-size:xx-small">{os.path.basename(filename)}</td> <td><audio controls src="data:audio/wav;base64,{encoded}"/></td><td><div><b>{reference}</b></div><div>{transcript}</div></td></tr>\n')

	vis.write('</tbody></table></body></html>')

def meanstd(logits):
	cov = lambda m: m @ m.t()
	L = torch.load(logits, map_location = 'cpu')

	batch_m = [b.mean(dim = -1) for b in L['features']]
	batch_mean = torch.stack(batch_m).mean(dim = 0)
	batch_s = [b.std(dim = -1) for b in L['features']]
	batch_std = torch.stack(batch_s).mean(dim = 0)
	batch_c = [cov(b - m.unsqueeze(-1)) for b, m in zip(L['features'], batch_m)]
	batch_cov = torch.stack(batch_c).mean(dim = 0)

	conv1_m = [b.mean(dim = -1) for b in L['conv1']]
	conv1_mean = torch.stack(conv1_m).mean(dim = 0)
	conv1_s = [b.std(dim = -1) for b in L['conv1']]
	conv1_std = torch.stack(conv1_s).mean(dim = 0)
	conv1_c = [cov(b - m.unsqueeze(-1)) for b, m in zip(L['conv1'], conv1_m)]
	conv1_cov = torch.stack(conv1_c).mean(dim = 0)
	
	plt.subplot(231)
	plt.imshow(batch_mean[:10].unsqueeze(0), origin = 'lower', aspect = 'auto')
	plt.subplot(232)
	plt.imshow(batch_std[:10].unsqueeze(0), origin = 'lower', aspect = 'auto')
	plt.subplot(233)
	plt.imshow(batch_cov[:20, :20], origin = 'lower', aspect = 'auto')

	plt.subplot(234)
	plt.imshow(conv1_mean.unsqueeze(0), origin = 'lower', aspect = 'auto')
	plt.subplot(235)
	plt.imshow(conv1_std.unsqueeze(0), origin = 'lower', aspect = 'auto')
	plt.subplot(236)
	plt.imshow(conv1_cov, origin = 'lower', aspect = 'auto')
	plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.8, wspace=0.4)
	plt.savefig(logits + '.jpg', dpi = 150)

def entropy(logits):
	L = torch.load(logits, map_location = 'cpu')
	entropy = lambda x, dim = 0, eps = 1e-16: -(x * (x + eps).log()).sum(dim = dim)
	e = [entropy(F.softmax(l, dim = 1), dim = 1).mean() for l in L['logits']]
	print(os.path.basename(logits), 'Entropy:', float(torch.tensor(e).mean()))

def cer(experiments_dir, experiment_id):
	res = collections.defaultdict(list)
	experiment_dir = os.path.join(experiments_dir, experiment_id)
	for f in sorted(glob.glob(os.path.join(experiment_dir, f'transcripts_*.json'))):
		eidx = f.find('epoch')
		iteration = f[eidx:].replace('.json', '')
		val_dataset_name = f[f.find('transcripts_') + len('transcripts_'):eidx]
		checkpoint = os.path.join(experiment_id, 'checkpoint_' + f[eidx:].replace('.json', '.pt'))
		cer = float(torch.tensor([j['cer'] for j in json.load(open(f))]).mean())
		res[iteration].append((val_dataset_name, cer, checkpoint))
	val_dataset_names = sorted(set(val_dataset_name for r in res.values() for val_dataset_name, cer, checkpoint in r))
	print('iteration\t' + '\t'.join(val_dataset_names) + '\tcheckpoint')
	for iteration, r in res.items():
		cers = {val_dataset_name : f'{cer:.04f}' for val_dataset_name, cer, checkpoint in r}
		print(f'{iteration}\t' + '\t'.join(cers.get(val_dataset_name, '') for val_dataset_name in val_dataset_names) + f'\t{r[-1][-1]}')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()

	cmd = subparsers.add_parser('tra')
	cmd.add_argument('--transcripts', default = 'data/transcripts.json')
	cmd.set_defaults(func = tra)

	cmd = subparsers.add_parser('meanstd')
	cmd.add_argument('--logits', default = 'data/logits.pt')
	cmd.set_defaults(func = meanstd)

	cmd = subparsers.add_parser('entropy')
	cmd.add_argument('--logits', default = 'data/logits.pt')
	cmd.set_defaults(func = entropy)

	cmd = subparsers.add_parser('cer')
	cmd.add_argument('experiment_id')
	cmd.add_argument('--experiments-dir', default = 'data/experiments')
	cmd.set_defaults(func = cer)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
