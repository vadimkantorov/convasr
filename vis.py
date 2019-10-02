import os
import collections
import glob
import json
import io
import argparse
import base64
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import dataset
import ru
import metrics
import models

labels = dataset.Labels(ru)

def errors(transcripts):
	ref_tra = list(sorted(json.load(open(transcripts)), key = lambda j: j['cer']))
	utterances = list(map(lambda j: metrics.analyze(j['reference'], j['transcript'], phonetic_replace_groups = ru.PHONETIC_REPLACE_GROUPS), ref_tra))
	full_form = {w.rstrip(ru.VOWELS) : w for utt in utterances for w in utt['words']['errors']}
	words = collections.Counter(w.rstrip(ru.VOWELS) for utt in utterances for w in utt['words']['errors'])
	json.dump(dict(utterances = utterances, words = {full_form[w] : cnt for w, cnt in words.items()}), open(transcripts + '.errors.json', 'w'), indent = 2, sort_keys = True, ensure_ascii = False)

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

def cer(experiments_dir, experiment_id, entropy, loss):
	if experiment_id.endswith('.json'):
		reftra = json.load(open(experiment_id))
		for reftra_ in reftra:
			hyp = labels.normalize_transcript(labels.normalize_text(reftra_['transcript']))
			ref = labels.normalize_transcript(labels.normalize_text(reftra_['reference']))
			reftra_['cer'] = metrics.cer(hyp, ref)
			reftra_['wer'] = metrics.wer(hyp, ref)
		cer_avg, wer_avg = [float(torch.tensor([r[k] for r in reftra]).mean()) for k in ['cer', 'wer']]
		print(f'CER: {cer_avg:.02f} | WER: {wer_avg:.02f}')
		return

	res = collections.defaultdict(list)
	experiment_dir = os.path.join(experiments_dir, experiment_id)
	for f in filter(lambda f: not f.endswith('.errors.json'), sorted(glob.glob(os.path.join(experiment_dir, f'transcripts_*.json')))):
		eidx = f.find('epoch')
		iteration = f[eidx:].replace('.json', '')
		val_dataset_name = f[f.find('transcripts_') + len('transcripts_'):eidx]
		checkpoint = os.path.join(experiment_id, 'checkpoint_' + f[eidx:].replace('.json', '.pt'))
		cer = torch.tensor([j['entropy' if entropy else 'loss' if loss else 'cer'] for j in json.load(open(f))] or [0.0])
		res[iteration].append((val_dataset_name, float(cer.mean()), checkpoint))
	val_dataset_names = sorted(set(val_dataset_name for r in res.values() for val_dataset_name, cer, checkpoint in r))
	print('iteration\t' + '\t'.join(val_dataset_names) + '\tcheckpoint')
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
	ticks = lambda labelsize = 3, length = 0: plt.gca().tick_params(axis='both', which='both', labelsize=labelsize, length=length) or [ax.set_linewidth(0) for ax in plt.gca().spines.values()]
	html = open(output_path + '.html', 'w')
	html.write('<html><body>') 
	for segment in torch.load(logits):
		audio_path, filename, features, logits = map(segment.get, ['audio_path', 'filename', 'features', 'logits'])
		log_probs = F.log_softmax(logits, dim = 0)
		entropy = models.entropy(log_probs, dim = 0, sum = False)
		margin = models.margin(log_probs, dim = 0)
		energy = features.exp().sum(dim = 0)[::2]

		plt.figure(figsize = (5, 0.7))
		plt.suptitle(filename, fontsize = 4)
		plt.plot(energy / energy.max(), 'b', linewidth = 0.3)
		plt.plot(entropy, 'r', linewidth = 0.3)
		plt.hlines(1.0, 0, entropy.shape[-1] - 1, linewidth = 0.5)
		bad = entropy > MAX_ENTROPY
		bad_ = torch.cat([bad[-1:], bad[:-1]])
		for begin, end in zip((bad & ~bad_).nonzero().squeeze(1).tolist(), (~bad & bad_).nonzero().squeeze(1).tolist()):
			plt.axvspan(begin, end, color='red', alpha=0.5)

		plt.ylim(0, 2)
		plt.xlim(0, entropy.shape[-1] - 1)
		plt.xticks(torch.arange(entropy.shape[-1]), labels.idx2str(log_probs.argmax(dim = 0), eps = '.', repeat = '_'))
		ticks()
		plt.subplots_adjust(left=0, right=1, top=1, bottom=0.2)
		buf = io.BytesIO()
		plt.savefig(buf, format = 'jpg', dpi = 300)
		plt.close()
		
		encoded = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
		html.write(f'<img style="width:100%" src="data:image/jpeg;base64,{encoded}"></img>')
		encoded = base64.b64encode(open(audio_path, 'rb').read()).decode('utf-8').replace('\n', '')
		html.write(f'<audio style="width:100%" controls src="data:audio/wav;base64,{encoded}"></audio><hr/>')
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

def checksegments(audio_path):
	encode_audio = lambda audio_path: base64.b64encode(open(audio_path, 'rb').read()).decode('utf-8').replace('\n', '')
	segments = list(sorted([j['begin'], j['end'], j['channel'], j['segment_path']] for j in json.load(open(audio_path + '.json'))))
	html = open(audio_path + '.html', 'w')
	html.write('<html><head><meta charset="UTF-8"><style>.channel0{background-color:violet} .channel1{background-color:lightblue} .reference{opacity:0.4} .on{background-color:green} .off{background-color:red}</style></head><body>')
	html.write(f'<h4>{os.path.basename(audio_path)}</h4>')
	html.write(f'<audio style="width:100%" controls src="data:audio/wav;base64,{encode_audio(audio_path)}"></audio>')
	html.write('<div>channel #0:&nbsp;<span></span></div><div>channel #1:&nbsp;<span></span></div>')
	html.write('<table><thead><tr><th>#</th><th>begin</th><th>end</th><th style="width:100%">segment</th></tr></thead><tbody>')
	html.write(''.join(f'<tr class="channel{c}"><td><strong>{c}</strong></td><td><a onclick="play({b:.02f}); return false;" href="#" target="_blank">{b:.02f}</a></td><td>{e:.02f}</td><td><audio style="width:100%" controls src="data:audio/wav;base64,{encode_audio(s)}"></audio></td></tr>' for b, e, c, s in segments))
	html.write('</tbody></table>')
	html.write('''<script>
		const segments = SEGMENTS;

		function play(time)
		{
			const audio = document.querySelector('audio');
			audio.currentTime = time;
			audio.play();
		};

		document.querySelector('audio').ontimeupdate = (evt) =>
		{
			const [div0, div1] = document.querySelectorAll('div');
			const [span0, span1] = document.querySelectorAll('span');
			const time = evt.target.currentTime;
			const [begin0, end0, channel0, segment_path0] = segments.find(([begin, end, channel, segment_path]) => channel == 0 && begin <= time && time <= end) || [null, null, 0, null];
			const [begin1, end1, channel1, segment_path1] = segments.find(([begin, end, channel, segment_path]) => channel == 1 && begin <= time && time <= end) || [null, null, 1, null];
			div0.className = begin0 ? 'on' : 'off';
			div1.className = begin1 ? 'on' : 'off';
			span0.innerText = begin0 ? `${begin0.toFixed(2)}-${end0.toFixed(2)}` : '';
			span1.innerText = begin1 ? `${begin1.toFixed(2)}-${end1.toFixed(2)}` : '';
		};
	</script>'''.replace('SEGMENTS', repr(segments)))
	html.write('</body></html>')

def parseslicing(slicing):
	by_audio_name = collections.defaultdict(list)
	for line in open(slicing):
		splitted = line.split('\t')
		audio_name = splitted[1]
		begin, end = splitted[0].split('_')[0].split('-')
		channel = splitted[2]
		by_audio_name[audio_name].append(dict(begin = float(begin), end = float(end), channel = int(channel), segment_path = os.path.join(os.path.dirname(slicing), splitted[0])))

	for audio_name, segments in by_audio_name.items():
		json.dump(segments, open(os.path.join(os.path.dirname(slicing), 'source', audio_name + '.json'), 'w'), indent = 2, sort_keys = True)

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
	cmd.set_defaults(func = cer)

	cmd = subparsers.add_parser('errors')
	cmd.add_argument('transcripts', default = 'data/transcripts.json')
	cmd.set_defaults(func = errors)

	cmd = subparsers.add_parser('words')
	cmd.add_argument('train_data_path')
	cmd.add_argument('val_data_path')
	cmd.set_defaults(func = words)

	cmd = subparsers.add_parser('vis')
	cmd.add_argument('logits')
	cmd.set_defaults(func = vis)

	cmd = subparsers.add_parser('checksegments')
	cmd.add_argument('audio_path')
	cmd.set_defaults(func = checksegments)

	cmd = subparsers.add_parser('parseslicing')
	cmd.add_argument('slicing')
	cmd.set_defaults(func = parseslicing)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
