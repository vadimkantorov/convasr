import os
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
		vis.write(f'<tr><td style="border-right: 2px black solid">{cer:.02%}</td> <td style="font-size:xx-small">{os.path.basename(filename)}</td> <td><audio controls src="data:audio/wav;base64,{encoded}"/></td><td><div><b>{reference}</b></div><div>{transcript}</div></td></tr>\n')

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

def cer(experiments_dir, id, val_dataset_name):
    experiment_dir = os.path.join(experiments_dir, id)
    for f in glob.glob(os.path.join(experiment_dir, f'transcripts_{val_dataset_name}*.json')):
        iteration = f[f.find('iter'):]
        checkpoint = os.path.abspath(os.path.join(experiment_dir, 'checkpoint_' + f[f.find('epoch'):].replace('.json', '.pt')))
        cer = float(torch.tensor([j['cer'] for j in json.load(open(f))]).mean())
        print(f'{iteration}    {cer:.02f}     {checkpoint}')

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
    cmd.add_argument('--experiments-dir')
    cmd.add_argument('--id')
    cmd.add_argument('--val-dataset-name')
    cmd.set_defaults(func = cer)

    args = vars(parser.parse_args())
    func = args.pop('func')
    func(**args)
