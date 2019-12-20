import os
import collections
import glob
import json
import io
import sys
import time
import random
import itertools
import argparse
import base64
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import altair
import torch
import torch.nn.functional as F
import dataset
import ru
import metrics
import models

labels = dataset.Labels(ru)

def histc_vega(tensor, min, max, bins):
	bins = torch.linspace(min, max, bins)
	hist = tensor.histc(min = bins.min(), max = bins.max(), bins = len(bins)).int()
	return altair.Chart(altair.Data(values = [dict(x = b, y = v) for b, v in zip(bins.tolist(), hist.tolist())])).mark_bar().encode(x = altair.X('x:Q'), y = altair.Y('y:Q')).to_dict()

def tra(transcripts):
	ref_tra = list(sorted(json.load(open(transcripts)), key = lambda j: j['cer']))
	vis = open(transcripts + '.html' , 'w')
	vis.write(f'<html><meta charset="utf-8"><body><h1>{args.transcripts}</h1><table style="border-collapse:collapse"><thead><tr><th>cer</th><th>filename</th><th>audio</th><th><div>reference</div><div>transcript</div></th></tr></thead><tbody>')

	for i, (reference, transcript, filename, cer) in enumerate(list(map(j.get, ['reference', 'transcript', 'filename', 'cer'])) for j in ref_tra):
		encoded = base64.b64encode(open(filename, 'rb').read()).decode()
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

def cer(experiments_dir, experiment_id, entropy, loss, cer10, cer15, cer20, cer30, cer40, cer50, per, json_, bpe):
	if experiment_id.endswith('.json'):
		reftra = json.load(open(experiment_id))
		for reftra_ in reftra:
			hyp = labels.postprocess_transcript(labels.normalize_text(reftra_.get('hyp', ref_tra_.get('transcript'))     ))
			ref = labels.postprocess_transcript(labels.normalize_text(reftra_.get('ref', ref_tra_.get('reference'))      ))
			reftra_['cer'] = metrics.cer(hyp, ref)
			reftra_['wer'] = metrics.wer(hyp, ref)

		cer_, wer_ = [torch.tensor([r[k] for r in reftra])for k in ['cer', 'wer']]
		cer_avg, wer_avg = float(cer_.mean()), float(wer_.mean())
		print(f'CER: {cer_avg:.02f} | WER: {wer_avg:.02f}')
		loss_ = torch.tensor([r.get('loss', 0) for r in reftra])
		loss_ = loss_[~(torch.isnan(loss_) | torch.isinf(loss_))]
		#min, max, steps = 0.0, 2.0, 20
		#bins = torch.linspace(min, max, steps = steps)
		#hist = torch.histc(loss_, bins = steps, min = min, max = max)
		#for b, h in zip(bins.tolist(), hist.tolist()):
		#	print(f'{b:.02f}\t{h:.0f}')

		plt.figure(figsize = (8, 4))
		plt.suptitle(os.path.basename(experiment_id))
		plt.subplot(211)
		plt.title('cer PDF')
		#plt.hist(cer_, range = (0.0, 1.2), bins = 20, density = True)
		seaborn.distplot(cer_, bins = 20, hist = True)
		plt.xlim(0, 1)
		plt.subplot(212)
		plt.title('cer CDF')
		plt.hist(cer_, bins = 20, density = True, cumulative = True)
		plt.xlim(0, 1)
		plt.xticks(torch.arange(0, 1.01, 0.1))
		plt.grid(True)

		#plt.subplot(223)
		#plt.title('loss PDF')
		#plt.hist(loss_, range = (0.0, 2.0), bins = 20, density = True)
		#seaborn.distplot(loss_, bins = 20, hist = True)
		#plt.xlim(0, 3)
		#plt.subplot(224)
		#plt.title('loss CDF')
		#plt.hist(loss_, bins = 20, density = True, cumulative = True)
		#plt.grid(True)
		plt.subplots_adjust(hspace = 0.4)
		plt.savefig(experiment_id + '.png', dpi = 150)
		return

	res = collections.defaultdict(list)
	experiment_dir = os.path.join(experiments_dir, experiment_id)
	for f in sorted(glob.glob(os.path.join(experiment_dir, f'transcripts_*.json'))):
		eidx = f.find('epoch')
		iteration = f[eidx:].replace('.json', '')
		val_dataset_name = f[f.find('transcripts_') + len('transcripts_'):eidx]
		checkpoint = os.path.join(experiment_dir, 'checkpoint_' + f[eidx:].replace('.json', '.pt')) if not json_ else f
		val = torch.tensor([j['entropy' if entropy else 'loss' if loss else 'per' if per else 'cer'] for j in json.load(open(f)) if j['labels'].startswith(bpe)] or [0.0])
		val = val[~(torch.isnan(val) | torch.isinf(val))]

		if cer10 or cer20 or cer30 or cer40 or cer50:
			val = (val < 0.1 * [False, cer10, cer20, cer30, cer40, cer50].index(True)).float()
		if cer15:
			val = (val < 0.15).float()

		res[iteration].append((val_dataset_name, float(val.mean()), checkpoint))
	val_dataset_names = sorted(set(val_dataset_name for r in res.values() for val_dataset_name, cer, checkpoint in r))
	print('iteration\t' + '\t'.join(val_dataset_names))
	for iteration, r in res.items():
		cers = {val_dataset_name : f'{cer:.04f}' for val_dataset_name, cer, checkpoint in r}
		print(f'{iteration}\t' + '\t'.join(cers.get(val_dataset_name, '') for val_dataset_name in val_dataset_names) + f'\t{r[-1][-1]}')

def words(train_data_path, val_data_path):
	train_cnt = collections.Counter(w for l in open(train_data_path) for w in l.split(',')[1].split())
	val_cnt = collections.Counter(w for l in open(val_data_path) for w in l.split(',')[1].split())

	for w, c1 in val_cnt.most_common():
		c2 = train_cnt[w]
		if c1 > 1 and c2 < 1000:
			print(w, c1, c2)

def vis(logits, MAX_ENTROPY = 1.0):
	ticks = lambda labelsize = 2.5, length = 0: plt.gca().tick_params(axis = 'both', which = 'both', labelsize = labelsize, length = length) or [ax.set_linewidth(0) for ax in plt.gca().spines.values()]
	logits_path = logits + '.html'
	html = open(logits_path, 'w')
	html.write('<html><body>')
	for segment in torch.load(logits):
		audio_path, filename, logits, log_probs, cer, reference_aligned, transcript_aligned = map(segment.get, ['audio_path', 'filename', 'logits', 'log_probs', 'cer', 'reference_aligned', 'transcript_aligned'])
		entropy = models.entropy(log_probs, dim = 0, sum = False)
		margin = models.margin(log_probs, dim = 0)
		#energy = features.exp().sum(dim = 0)[::2]

		plt.figure(figsize = (6, 0.7))
		plt.suptitle(filename, fontsize = 4)
		top1, top2 = log_probs.exp().topk(2, dim = 0).values
		plt.hlines(1.0, 0, entropy.shape[-1] - 1, linewidth = 0.2)
		plt.plot(top1, 'b', linewidth = 0.3)
		plt.plot(top2, 'g', linewidth = 0.3)
		plt.plot(entropy, 'r', linewidth = 0.3)
		bad = entropy > MAX_ENTROPY
		bad_ = torch.cat([bad[1:], bad[:1]])
		for begin, end in zip((~bad & bad_).nonzero().squeeze(1).tolist(), (bad & ~bad_).nonzero().squeeze(1).tolist()):
			plt.axvspan(begin, end, color='red', alpha=0.2)

		plt.ylim(0, 2)
		plt.xlim(0, entropy.shape[-1] - 1)
		xlabels = list(map('\n'.join, zip(*labels.idx2str(log_probs.topk(5, dim = 0).indices, blank = '.', space = '_'))))
		#xlabels = labels.idx2str(log_probs.argmax(dim = 0)).replace(labels.blank, '.').replace(labels.space, '_')
		plt.xticks(torch.arange(entropy.shape[-1]), xlabels, fontfamily = 'monospace')
		ticks()
		
		plt.subplots_adjust(left=0, right=1, top=1, bottom=0.4)
		buf = io.BytesIO()
		plt.savefig(buf, format = 'jpg', dpi = 600)
		plt.close()
		
		html.write(f'<h4>{audio_name} | cer: {cer:.02f}</h4>')
		html.write(f'<pre>ref: {ref_aligned}</pre>')
		html.write(f'<pre>hyp: {hyp_aligned}</pre>')
		html.write('<img style="width:100%" src="data:image/jpeg;base64,{encoded}"></img>'.format(encoded = base64.b64encode(buf.getvalue()).decode()))	
		html.write('<audio style="width:100%" controls src="data:audio/wav;base64,{encoded}"></audio><hr/>'.format(encoded = base64.b64encode(open(audio_path, 'rb').read()).decode()))
	html.write('''<script>
		Array.from(document.querySelectorAll('img')).map(img => {
			img.onclick = (evt) => {
				const img = evt.target;
				const dim = img.getBoundingClientRect();
				const x = (evt.clientX - dim.left) / dim.width;
				const audio = img.nextSibling;
				audio.currentTime = x * audio.duration;
				audio.play();
			};
		});
	</script>''')
	html.write('</body></html>')
	print('\n', logits_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()

	cmd = subparsers.add_parser('tra')
	cmd.add_argument('transcripts', default = 'data/transcripts.json')
	cmd.set_defaults(func = tra)

	cmd = subparsers.add_parser('meanstd')
	cmd.add_argument('--logits', default = 'data/logits.pt')
	cmd.set_defaults(func = meanstd)

	cmd = subparsers.add_parser('cer')
	cmd.add_argument('experiment_id')
	cmd.add_argument('--experiments-dir', default = 'data/experiments')
	cmd.add_argument('--entropy', action = 'store_true')
	cmd.add_argument('--loss', action = 'store_true')
	cmd.add_argument('--per', action = 'store_true')
	cmd.add_argument('--cer10', action = 'store_true')
	cmd.add_argument('--cer15', action = 'store_true')
	cmd.add_argument('--cer20', action = 'store_true')
	cmd.add_argument('--cer30', action = 'store_true')
	cmd.add_argument('--cer40', action = 'store_true')
	cmd.add_argument('--cer50', action = 'store_true')
	cmd.add_argument('--json', dest = "json_", action = 'store_true')
	cmd.add_argument('--bpe', default = 'char')
	cmd.set_defaults(func = cer)

	cmd = subparsers.add_parser('words')
	cmd.add_argument('train_data_path')
	cmd.add_argument('val_data_path')
	cmd.set_defaults(func = words)

	cmd = subparsers.add_parser('vis')
	cmd.add_argument('logits')
	cmd.set_defaults(func = vis)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
