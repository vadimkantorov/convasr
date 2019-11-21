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
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn
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
	for f in filter(lambda f: not f.endswith('.errors.json'), sorted(glob.glob(os.path.join(experiment_dir, f'transcripts_*.json')))):
		eidx = f.find('epoch')
		iteration = f[eidx:].replace('.json', '')
		val_dataset_name = f[f.find('transcripts_') + len('transcripts_'):eidx]
		checkpoint = os.path.join(experiment_dir, 'checkpoint_' + f[eidx:].replace('.json', '.pt')) if not json_ else f
		val = torch.tensor([j['entropy' if entropy else 'loss' if loss else 'per' if per else 'cer'] for j in json.load(open(f)) if j['labels'] == ('bpe' if bpe else 'char') ] or [0.0])
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

def checksegments(audio_path):
	encode_audio = lambda audio_path: base64.b64encode(open(audio_path, 'rb').read()).decode()
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

def exphtml(root_dir, html_dir = 'public', strftime = '%Y-%m-%d %H:%M:%S'):
	#TODO: group by experiment_id, group by iteration, columns x columns
	json_dir = os.path.join(root_dir, 'json')
	html_dir = os.path.join(root_dir, html_dir)
	os.makedirs(html_dir, exist_ok = True)
	html_path = os.path.join(html_dir, 'index.html')

	def json_load(path):
		try:
			return json.load(open(path))
		except:
			return {}

	jsons = list(filter(None, (json_load(os.path.join(json_dir, json_file)) for json_file in os.listdir(json_dir))))

	by_experiment_key = lambda j: j['experiment_id']
	by_time_key = lambda j: j['time']
	by_iteration = lambda j: (j['iteration'], j['time'])
	by_time_last_key = lambda experiment: by_time_key(experiment[0][-1])
	fields_or_default = lambda j: dict(default = j) if not isinstance(j, dict) else j

	experiments = [(list(g), k) for k, g in itertools.groupby(sorted(jsons, key = by_experiment_key), key = by_experiment_key)]
	experiments = list(sorted(( (list(sorted(g, key = by_iteration)), k) for g, k in experiments), key = by_time_last_key, reverse = True))

	columns = list(sorted(set(c for g, *_ in experiments for j in g for c in j['columns'])))
	fields = list(sorted(set(f for g, *_ in experiments for j in g for c in j['columns'].values() for f in fields_or_default(c))))
	field = fields[0]
	
	generated_time = time.strftime(strftime, time.gmtime())
	fmt = lambda o: '{:.04f}'.format(o) if isinstance(o, float) else str(o)

	with open(html_path, 'w') as html:
		html.write(f'<html><head><title>Resutls</title></head><body>\n')
		html.write('<script>var toggle = className => Array.from(document.querySelectorAll(`.${className}`)).map(e => {e.hidden = !e.hidden});</script>')
		html.write('<table cellpadding="2px" cellspacing="0">')
		html.write(f'<h1>Generated at {generated_time}</h1>')
		html.write('<div>fields:' + ''.join(f'<input type="checkbox" name="{f}" value="field{hash(f)}" {"checked" if f == field else ""} onchange="toggle(event.target.value)""><label for="{f}">{f}</label>' for f in fields) + '</div>\n')
		html.write('<div>columns:' + ''.join(f'<input type="checkbox" name="{c}" value="col{hash(c)}" checked onchange="toggle(event.target.value)"><label for="{c}">{c}</label>' for c in columns) + '</div>\n')
		html.write('<hr />')
		for jsons, experiment_id in experiments:
			idx = set([0, len(jsons) - 1] + [i for i, j in enumerate(jsons) if 'iter' not in j['iteration']])

			generated_time = time.strftime(strftime, time.localtime(jsons[-1]['time']))
			html.write(f'''<tr><td title="{generated_time}" onclick="toggle('{experiment_id}.hidden')"><strong>{experiment_id}</strong></td>''' + ''.join(f'<td class="col{hash(c)}"><strong>{c}</strong></td>' for c in columns) + '</tr>')
			for i, j in enumerate(jsons):
				j['git_http'] = j.get('git_http', '')
				j['git_revision'] = j.get('git_revision', '')
				j['git_comment'] = j.get('git_comment', '')
				generated_time = time.strftime(strftime, time.localtime(j['time']))
				hidden = 'hidden' if i not in idx else ''
				meta_key = f'meta{hash(experiment_id + str(j["iteration"]))}'
				meta = json.dumps(j['meta'], sort_keys = True, indent = 2, ensure_ascii = False) if j.get('meta') else None
				html.write(f'<tr class="{hidden} {experiment_id}" {hidden}>')
				html.write(f'''<td onclick="toggle('{meta_key}')" title="{generated_time}" style="border-right: 1px solid black">{j["iteration"]}</td>''')
				html.write(''.join(f'<td class="col{hash(c)}">' + ''.join(f'<span style="margin-right:3px" {"hidden" if f != field else ""} class="field{hash(f)}">{fmt(j["columns"].get(c, {}).get(f, ""))}</span>' for f in fields) + '</td>' for c in columns))
				html.write('</tr>\n')
				html.write('<tr hidden class="{meta_key}" style="background-color:lightgray"><td><a href="{git_http}">@{git_revision}</a></td><td colspan="100">{git_comment}</td></tr>\n'.format(meta_key = meta_key, **j))
				html.write(f'<tr hidden class="{meta_key}" style="background-color:lightgray"><td colspan="100"><pre>{meta}</pre></td></tr>\n' if meta else '')

			html.write('<tr><td>&nbsp;</td></tr>')
		html.write('</table></body></html>')

	try:
		print('Committing updated vis at ', html_path)
		subprocess.check_call(['git', 'pull'], cwd = root_dir)
		subprocess.check_call(['git', 'add', '-A'], cwd = root_dir)
		subprocess.check_call(['git', 'commit', '-a', '--allow-empty-message', '-m', ''], cwd = root_dir)
		subprocess.check_call(['git', 'push'], cwd = root_dir)
	except:
		print(sys.exc_info())

def expjson(root_dir, experiment_id, epoch = None, iteration = None, columns = {}, meta = {}, name = None, git_revision = True, git_http = None):
	if git_revision is True:
		try:
			git_revision, git_comment = map(lambda b: b.decode('utf-8'), subprocess.check_output(['git', 'log', '--format=%h%x00%s', '--no-decorate', '-1']).split(b'\x00'))
		except:
			git_revision, git_comment = 'error', 'error'
	else:
		git_revision, git_comment = ''

	obj = dict(experiment_id = experiment_id, iteration = f'epoch{epoch:02d}_iter{iteration:07d}' if epoch is not None and iteration is not None else 'test', columns = columns, time = int(time.time()), meta = meta, git_revision = git_revision, git_comment = git_comment, git_http = git_http.replace('%h', git_revision) if git_http else None)
	
	json_dir = os.path.join(root_dir, 'json')
	os.makedirs(json_dir, exist_ok = True)
	name = f'{int(time.time())}.{random.randint(10, 99)}.json' if name is None else name
	json_path = os.path.join(json_dir, name)
	json.dump(obj, open(json_path, 'w'), sort_keys = True, indent = 2, ensure_ascii = False)

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
	cmd.add_argument('--bpe', action = 'store_true')
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

	cmd = subparsers.add_parser('exphtml')
	cmd.set_defaults(func = exphtml)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
