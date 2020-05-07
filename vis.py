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
import random
import matplotlib.pyplot as plt
import seaborn
import altair
import torch
import torch.nn.functional as F
import audio
import tools
import datasets
import decoders
import metrics
import models
import ctc
import transcripts
import ru as lang

def transcript(html_path, sample_rate, mono, transcript, filtered_transcript = []):
	if isinstance(transcript, str):
		transcript = json.load(open(transcript))

	has_hyp = True
	has_ref = any(t['alignment']['ref'] for t in transcript)

	audio_path = transcript[0]['audio_path']
	signal, sample_rate = audio.read_audio(audio_path, sample_rate = sample_rate, mono = mono, normalize = False)
	
	fmt_link = lambda ref = '', hyp = '', channel = 0, begin = 0, end = 0, speaker = '', i = '', j = '': (f'<a onclick="return play({channel},{begin},{end})"' if ref not in [0, 1] else '<span') + f' title="#{channel}. {speaker}: {begin:.04f} - {end:.04f} | {i} - {j}" href="#" target="_blank">' + ((ref + hyp) if isinstance(ref, str) else f'{begin:.02f}' if ref == 0 else f'{end:.02f}' if ref == 1 else f'{end - begin:.02f}') + ('</a>' if ref not in [0, 1] else '</span>')
	fmt_words = lambda rh: ' '.join(fmt_link(**w) for w in rh)
	fmt_begin_end = 'data-begin="{begin}" data-end="{end}"'.format

	html = open(html_path, 'w')
	html.write('<html><head><meta charset="UTF-8"><style>a {text-decoration: none;} .channel0 .hyp{padding-right:150px} .channel1 .hyp{padding-left:150px}     .ok{background-color:green} .m0{margin:0px} .top{vertical-align:top} .channel0{background-color:violet} .channel1{background-color:lightblue;' + ('display:none' if len(signal) == 1 else '') + '} .reference{opacity:0.4} .channel{margin:0px}</style></head><body>')
	html.write(f'<div style="overflow:auto"><h4 style="float:left">{os.path.basename(audio_path)}</h4><h5 style="float:right">0.000000</h5></div>')
	html.writelines(f'<figure class="m0"><figcaption>channel #{c}:</figcaption><audio ontimeupdate="ontimeupdate_(event)" onpause="onpause_(event)" id="audio{c}" style="width:100%" controls src="data:audio/wav;base64,{base64.b64encode(wav).decode()}"></audio></figure>' for c, wav in enumerate(audio.write_audio(io.BytesIO(), signal[channel], sample_rate).getvalue() for channel in ([0, 1] if len(signal) == 2 else []) + [...]))
	html.write(f'<pre class="channel"><h3 class="channel0 channel">hyp #0:<span class="subtitle"></span></h3></pre><pre class="channel"><h3 class="channel0 reference channel">ref #0:<span class="subtitle"></span></h3></pre><pre class="channel" style="margin-top: 10px"><h3 class="channel1 channel">hyp #1:<span class="subtitle"></span></h3></pre><pre class="channel"><h3 class="channel1 reference channel">ref #1:<span class="subtitle"></span></h3></pre><hr/><table style="width:100%">')
	html.write('<tr>' + ('<th>begin</th><th>end</th><th>dur</th><th style="width:50%">hyp</th>' if has_hyp else '') + ('<th style="width:50%">ref</th><th>begin</th><th>end</th><th>dur</th><th>cer</th>' if has_ref else '') + '<th>speaker</th></tr>')
	html.writelines(f'<tr class="channel{c}">'+ (f'<td class="top">{fmt_link(0, **transcripts.summary(hyp, ij = True))}</td><td class="top">{fmt_link(1, **transcripts.summary(hyp, ij = True))}</td><td class="top">{fmt_link(2, **transcripts.summary(hyp, ij = True))}</td><td class="top hyp" data-channel="{c}" {fmt_begin_end(**transcripts.summary(hyp, ij = True))}>{fmt_words(hyp)}<template>{word_alignment(t["words"], tag = "", hyp = True)}</template></td>' if has_hyp else '') + (f'<td class="top reference ref" data-channel="{c}" {fmt_begin_end(**transcripts.summary(ref, ij = True))}>{fmt_words(ref)}<template>{word_alignment(t["words"], tag = "", ref = True)}</template></td><td class="top">{fmt_link(0, **transcripts.summary(ref, ij = True))}</td><td class="top">{fmt_link(1, **transcripts.summary(ref, ij = True))}</td><td class="top">{fmt_link(2, **transcripts.summary(ref, ij = True))}</td><td class="top">{t["cer"]:.2%}</td>' if ref else ('<td></td>' * 5 if has_ref else '')) + f'''<td class="top {ok and 'ok'}">{speaker}</td></tr>''' for t in transcripts.sort(transcript) for ok in [t in filtered_transcript] for c, speaker, ref, hyp in [(t['channel'], t.get('speaker', '') or 'N/A', t['alignment']['ref'], t['alignment']['hyp'])] )
	html.write('''</tbody></table><script>
		function play(channel, begin, end)
		{
			Array.from(document.querySelectorAll('audio')).map(audio => audio.pause());
			const audio = document.querySelector(`#audio${channel}`);
			audio.currentTime = begin;
			audio.dataset.endTime = end;
			audio.play();
			return false;
		}
		
		function subtitle(segments, time, channel)
		{
			return (segments.find(([rh, c, b, e]) => c == channel && b <= time && time <= e ) || ['', channel, null, null])[0];
		}

		function onpause_(evt)
		{
			evt.target.dataset.endTime = null;
		}

		function ontimeupdate_(evt)
		{
			const time = evt.target.currentTime, endtime = evt.target.dataset.endTime;
			if(endtime && time > endtime)
				return evt.target.pause();

			document.querySelector('h5').innerText = time.toString();
			const [spanhyp0, spanref0, spanhyp1, spanref1] = document.querySelectorAll('span.subtitle');
			[spanhyp0.innerHTML, spanref0.innerHTML, spanhyp1.innerHTML, spanref1.innerHTML] = [subtitle(hyp_segments, time, 0), subtitle(ref_segments, time, 0), subtitle(hyp_segments, time, 1), subtitle(ref_segments, time, 1)];
		}

		const make_segment = td => [td.querySelector('template').innerHTML, td.dataset.channel, td.dataset.begin, td.dataset.end];
		const hyp_segments = Array.from(document.querySelectorAll('.hyp')).map(make_segment), ref_segments = Array.from(document.querySelectorAll('.ref')).map(make_segment);
	</script></body></html>''')
	print(html_path)

def logits(logits, audio_name, MAX_ENTROPY = 1.0):
	good_audio_name = set(map(str.strip, open(audio_name[0])) if os.path.exists(audio_name[0]) else audio_name)
	labels = datasets.Labels(ru)
	decoder = decoders.GreedyDecoder()
	tick_params = lambda ax, labelsize = 2.5, length = 0, **kwargs: ax.tick_params(axis = 'both', which = 'both', labelsize = labelsize, length = length, **kwargs) or [ax.set_linewidth(0) for ax in ax.spines.values()]
	logits_path = logits + '.html'
	html = open(logits_path, 'w')
	html.write('''<html><meta charset="utf-8"/><body><script>
		function onclick_(evt)
		{
			const img = evt.target;
			const dim = img.getBoundingClientRect();
			const t = (evt.clientX - dim.left) / dim.width;
			const audio = img.nextSibling;
			audio.currentTime = t * audio.duration;
			audio.play();
		}
	</script>''')
	for r in torch.load(logits):
		logits = r['logits']
		if good_audio_name and r['audio_name'] not in good_audio_name:
			continue
		
		ref_aligned, hyp_aligned = r['alignment']['ref'], r['alignment']['hyp']
		
		log_probs = F.log_softmax(logits, dim = 0)
		entropy = models.entropy(log_probs, dim = 0, sum = False)
		log_probs_ = F.log_softmax(logits[:-1], dim = 0)
		entropy_ = models.entropy(log_probs_, dim = 0, sum = False)
		margin = models.margin(log_probs, dim = 0)
		#energy = features.exp().sum(dim = 0)[::2]

		alignment = ctc.alignment(log_probs.unsqueeze(0).permute(2, 0, 1), r['y'].unsqueeze(0).long(), torch.LongTensor([log_probs.shape[-1]]), torch.LongTensor([len(r['y'])]), blank = len(log_probs) - 1).squeeze(0)

		plt.figure(figsize = (6, 2))
		
		prob_top1, prob_top2 = log_probs.exp().topk(2, dim = 0).values
		plt.hlines(1.0, 0, entropy.shape[-1] - 1, linewidth = 0.2)
		artist_prob_top1, = plt.plot(prob_top1, 'b', linewidth = 0.3)
		artist_prob_top2, = plt.plot(prob_top2, 'g', linewidth = 0.3)
		artist_entropy, = plt.plot(entropy, 'r', linewidth = 0.3)
		artist_entropy_, = plt.plot(entropy_, 'yellow', linewidth = 0.3)
		plt.legend([artist_entropy, artist_entropy_, artist_prob_top1, artist_prob_top2], ['entropy', 'entropy, no blank', 'top1 prob', 'top2 prob'], loc = 1, fontsize = 'xx-small', frameon = False)
		bad = (entropy > MAX_ENTROPY).tolist()
		#runs = []
		#for i, b in enumerate(bad):
		#	if b:
		#		if not runs or not bad[i - 1]:
		#			runs.append([i, i])
		#		else:
		#			runs[-1][1] += 1
		#for begin, end in runs:
		#	plt.axvspan(begin, end, color='red', alpha=0.2)

		plt.ylim(0, 3.0)
		plt.xlim(0, entropy.shape[-1] - 1)

		decoded = decoder.decode(log_probs.unsqueeze(0), K = 5)[0]
		xlabels = list(map('\n'.join, zip(*[labels.decode(d, replace_blank = '.', replace_space = '_', replace_repeat = False) for d in decoded])))
		#xlabels_ = labels.decode(log_probs.argmax(dim = 0).tolist(), blank = '.', space = '_', replace2 = False)
		plt.xticks(torch.arange(entropy.shape[-1]), xlabels, fontfamily = 'monospace')
		tick_params(plt.gca())

		ax = plt.gca().secondary_xaxis('top')
		ref, ref_ = labels.decode(r['y'].tolist(), replace_blank = '.', replace_space = '_', replace_repeat = False), alignment
		ax.set_xticklabels(ref)
		ax.set_xticks(ref_)
		tick_params(ax, colors = 'red')

		#k = 0
		#for i, c in enumerate(ref + ' '):
		#	if c == ' ':
		#		plt.axvspan(ref_[k] - 1, ref_[i - 1] + 1, facecolor = 'gray', alpha = 0.2)
		#		k = i + 1

		plt.subplots_adjust(left = 0, right = 1, bottom = 0.12, top = 0.95)

		buf = io.BytesIO()
		plt.savefig(buf, format = 'jpg', dpi = 600)
		plt.close()
		
		html.write('<h4>{audio_name} | cer: {cer:.02f}</h4>'.format(**r))
		html.write(word_alignment(r['words']))
		html.write('<img onclick="onclick_(event)" style="width:100%" src="data:image/jpeg;base64,{encoded}"></img>'.format(encoded = base64.b64encode(buf.getvalue()).decode()))	
		html.write('<audio style="width:100%" controls src="data:audio/wav;base64,{encoded}"></audio><hr/>'.format(encoded = base64.b64encode(open(r['audio_path'], 'rb').read()).decode()))
	html.write('</body></html>')
	print('\n', logits_path)

def errors(input_path, include = [], exclude = [], audio = False, output_file_name = None, sortdesc = None, topk = None, duration = None, cer = None, wer = None, mer = None, filter_transcripts = None):
	include, exclude = (sum([open(file_path).read().splitlines() for file_path in clude], []) for clude in [include, exclude])
	read_transcript = lambda path: list(filter(lambda r: (not include or r['audio_name'] in include) and (not exclude or r['audio_name'] not in exclude), json.load(open(path)) if isinstance(path, str) else path)) if path is not None else []
	ours, theirs = transcripts.prune(read_transcript(input_path[0]), duration = duration, cer = cer, wer = wer, mer = mer), [{r['audio_name'] : r for r in read_transcript(transcript)} for transcript in input_path[1:]]
	if filter_transcripts is None:
		if sortdesc is not None:
			filter_transcripts = lambda cat: list(sorted(cat, key = lambda utt: utt[0][sortdesc], reverse = True))
		else:
			filter_transcripts = lambda cat: cat

	cat = filter_transcripts([[a] + list(filter(None, [t.get(a['audio_name'], None) for t in theirs])) for a in ours])[slice(topk)]
				
	# TODO: add sorting https://stackoverflow.com/questions/14267781/sorting-html-table-with-javascript
	output_file_name = output_file_name or (input_path[0] + (include[0].split('subset')[-1] if include else '') + '.html')
	
	f = open(output_file_name , 'w')
	f.write('<html><meta charset="utf-8"><style>audio {width:100%} .br{border-right:2px black solid} tr.first>td {border-top: 1px solid black} tr.any>td {border-top: 1px dashed black}  .nowrap{white-space:nowrap}</style>')
	f.write('<body><table style="border-collapse:collapse; width: 100%"><tr><th></th><th>cer_easy</th><th>cer</th><th>wer</th><th>mer</th><th></th></tr>')
	f.write('<tr><td><strong>averages<strong></td></tr>')
	f.write('\n'.join('<tr><td class="br">{input_name}</td><td>{cer_easy:.02%}</td><td>{cer:.02%}</td><td>{wer:.02%}</td><td>{mer:.02%}</td></tr>'.format(input_name = os.path.basename(input_path[i]), cer_easy = metrics.nanmean(c, 'cer_easy'), cer = metrics.nanmean(c, 'cer'), wer = metrics.nanmean(c, 'wer'), mer = metrics.nanmean(c, 'mer')) for i, c in enumerate(zip(*cat))))
	f.write('<tr><td>&nbsp;</td></tr>')
	f.write('\n'.join(f'''<tr class="first"><td colspan="4">''' + (f'<audio controls src="data:audio/wav;base64,{base64.b64encode(open(utt[0]["audio_path"], "rb").read()).decode()}"></audio>' if audio else '') + f'<div class="nowrap">{utt[0]["audio_name"]}</div></td><td>{word_alignment(utt[0], ref = True, flat = True)}</td><td>{word_alignment(utt[0], ref = True, flat = True)}</td></tr>' + '\n'.join(f'<tr class="any"><td class="br">{os.path.basename(input_path[i])}</td><td>{a["cer_easy"]:.02%}</td><td>{a["cer"]:.02%}</td><td>{a["wer"]:.02%}</td><td class="br">{a["mer"]:.02%}</td><td>{word_alignment(a["words"])}</td><td>{word_alignment(a, hyp = True, flat = True)}</td></tr>' for i, a in enumerate(utt)) for utt in cat))
	f.write('</table></body></html>')
	print(output_file_name)

def audiosample(input_path, output_path, K):
	transcript = json.load(open(input_path))

	group = lambda t: t.get('group', 'group not found')
	by_group = {k : list(g) for k, g in itertools.groupby(sorted(transcript, key = group), key = group)}
	
	f = open(output_path, 'w')
	f.write('<html><meta charset="UTF-8"><body>')
	for group, transcript in sorted(by_group.items()):
		f.write(f'<h1>{group}</h1>')
		f.write('<table>')
		random.seed(1)
		random.shuffle(transcript)
		for t in transcript[:K]:
			try:
				encoded = base64.b64encode(open(os.path.join(args.dataset_root, t['audio_path']), 'rb').read()).decode()
			except:
				f.write('<tr><td>file not found: {audio_path}</td></tr>'.format(**t))
				continue
			f.write('<tr><td>{audio_path}</td><td><audio controls src="data:audio/wav;base64,{encoded}"/></td><td>{ref}</td></tr>\n'.format(encoded = encoded, **t))
		f.write('</table>')

	print(output_path)

def summary(input_path):
	transcript = json.load(open(input_path))
	cer_, wer_ = [torch.tensor([t[k] for t in transcript]) for k in ['cer', 'wer']]
	cer_avg, wer_avg = float(cer_.mean()), float(wer_.mean())
	print(f'CER: {cer_avg:.02f} | WER: {wer_avg:.02f}')

	loss_ = torch.tensor([t.get('loss', 0) for t in transcript])
	loss_ = loss_[~(torch.isnan(loss_) | torch.isinf(loss_))]
	#min, max, steps = 0.0, 2.0, 20
	#bins = torch.linspace(min, max, steps = steps)
	#hist = torch.histc(loss_, bins = steps, min = min, max = max)
	#for b, h in zip(bins.tolist(), hist.tolist()):
	#	print(f'{b:.02f}\t{h:.0f}')

	plt.figure(figsize = (8, 4))
	plt.suptitle(os.path.basename(input_path))
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
	plt.savefig(input_path + '.png', dpi = 150)

def tabulate(experiments_dir, experiment_id, entropy, loss, cer10, cer15, cer20, cer30, cer40, cer50, per, wer, json_, bpe, der):
	labels = datasets.Labels(lang)

	res = collections.defaultdict(list)
	experiment_dir = os.path.join(experiments_dir, experiment_id)
	for f in sorted(glob.glob(os.path.join(experiment_dir, f'transcripts_*.json'))):
		eidx = f.find('epoch')
		iteration = f[eidx:].replace('.json', '')
		val_dataset_name = f[f.find('transcripts_') + len('transcripts_'):eidx]
		checkpoint = os.path.join(experiment_dir, 'checkpoint_' + f[eidx:].replace('.json', '.pt')) if not json_ else f
		metric = 'wer' if wer else 'entropy' if entropy else 'loss' if loss else 'per' if per else 'der' if der else 'cer'
		val = torch.tensor([j[metric] for j in json.load(open(f)) if j['labels'].startswith(labels.alphabet)] or [0.0])
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

def histc_vega(tensor, min, max, bins):
	bins = torch.linspace(min, max, bins)
	hist = tensor.histc(min = bins.min(), max = bins.max(), bins = len(bins)).int()
	return altair.Chart(altair.Data(values = [dict(x = b, y = v) for b, v in zip(bins.tolist(), hist.tolist())])).mark_bar().encode(x = altair.X('x:Q'), y = altair.Y('y:Q')).to_dict()

def word_alignment(transcript, ref = None, hyp = None, flat = False, tag = '<pre>', prefix = True):
	span = lambda word, t = None: '<span style="{style}" title="{style or ''}">{word}</span>'.format(word = word, style = 'background-color:' + dict(ok = 'green', missing = 'red', missing_ref = 'darkred', typo_easy = 'lightgreen', typo_hard = 'pink')[t] if t is not None else '')
	
	if flat:
		ref_ = transcript.get('ref', '')
		hyp_ = transcript.get('hyp', '')
	else:
		ref_ = ' '.join(span(w['ref'], w['type'] if w['type'] == 'ok' else None) for w in transcript)
		hyp_ = ' '.join(span(w['hyp'], w['type']) for w in transcript)
	
	ref_ = ('ref: ' if prefix else '') + ref_
	hyp_ = ('hyp: ' if prefix else '') + hyp_
	contents = '\n'.join([ref_] if ref is True else [hyp_] if hyp is True else [ref_, hyp_])
	return tag + contents + tag.replace('<', '</')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()

	cmd = subparsers.add_parser('transcript')
	cmd.add_argument('--transcript', '-i')
	cmd.add_argument('--mono', action = 'store_true')
	cmd.add_argument('--sample-rate', type = int, default = 8_000)
	cmd.add_argument('--html-path', '-o')
	cmd.set_defaults(func = transcript)

	cmd = subparsers.add_parser('errors')
	cmd.add_argument('input_path', nargs = '+', default = ['data/transcripts.json'])
	cmd.add_argument('--include', nargs = '*', default = [])
	cmd.add_argument('--exclude', nargs = '*', default = [])
	cmd.add_argument('--audio', action = 'store_true')
	cmd.add_argument('--output-file-name', '-o')
	cmd.add_argument('--sortdesc', choices = ['cer', 'wer', 'mer', 'cer_easy'])
	cmd.add_argument('--topk', type = int)
	parser.add_argument('--cer', type = transcripts.number_tuple)
	parser.add_argument('--wer', type = transcripts.number_tuple)
	parser.add_argument('--mer', type = transcripts.number_tuple)
	parser.add_argument('--duration', type = transcripts.number_tuple)
	cmd.set_defaults(func = errors)

	cmd = subparsers.add_parser('tabulate')
	cmd.add_argument('experiment_id')
	cmd.add_argument('--experiments-dir', default = 'data/experiments')
	cmd.add_argument('--entropy', action = 'store_true')
	cmd.add_argument('--loss', action = 'store_true')
	cmd.add_argument('--per', action = 'store_true')
	cmd.add_argument('--wer', action = 'store_true')
	cmd.add_argument('--cer10', action = 'store_true')
	cmd.add_argument('--cer15', action = 'store_true')
	cmd.add_argument('--cer20', action = 'store_true')
	cmd.add_argument('--cer30', action = 'store_true')
	cmd.add_argument('--cer40', action = 'store_true')
	cmd.add_argument('--cer50', action = 'store_true')
	cmd.add_argument('--json', dest = "json_", action = 'store_true')
	cmd.add_argument('--bpe', default = 'char')
	cmd.add_argument('--der', action = 'store_true')
	cmd.set_defaults(func = tabulate)

	cmd = subparsers.add_parser('summary')
	cmd.add_argument('input_path')
	cmd.set_defaults(func = summary)

	cmd = subparsers.add_parser('words')
	cmd.add_argument('train_data_path')
	cmd.add_argument('val_data_path')
	cmd.set_defaults(func = words)

	cmd = subparsers.add_parser('logits')
	cmd.add_argument('logits')
	cmd.add_argument('--audio-name', nargs = '*')
	cmd.set_defaults(func = logits)

	cmd = subparsers.add_parser('audiosample')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o', required = True)
	cmd.add_argument('--dataset-root', default = '')
	cmd.add_argument('-K', type = int, default = 10)
	cmd.set_defaults(func = audiosample)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
