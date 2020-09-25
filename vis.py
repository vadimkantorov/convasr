import os
import collections
import glob
import json
import io
import sys
import math
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

channel_colors = ['gray', 'red', 'blue']

def speaker_barcode_img(transcript, begin, end, colors = channel_colors):
	assert begin == 0
	plt.figure(figsize = (8, 0.2))
	plt.xlim(begin, end)
	plt.yticks([])
	plt.axis('off')
	for t in transcript:
		plt.axvspan(t['begin'], t['end'], color = colors[t.get('speaker', 0)])
	plt.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
	buf = io.BytesIO()
	plt.savefig(buf, format = 'jpg', dpi = 150, facecolor = colors[0])
	plt.close()
	uri_speaker_barcode = base64.b64encode(buf.getvalue()).decode()
	return f'<img onclick="onclick_img(event)" src="data:image/jpeg;base64,{uri_speaker_barcode}" style="width:100%"></img>'

def speaker_barcode_svg(transcript, begin, end, colors = channel_colors, max_segment_seconds = 60):
	html = ''
	segments = transcripts.segment(transcript, max_segment_seconds = max_segment_seconds, break_on_speaker_change = False, break_on_channel_change = False)
	for segment in segments:
		summary = transcripts.summary(segment)
		duration = transcripts.compute_duration(summary)
		if duration <= max_segment_seconds:
			duration = max_segment_seconds
		header = '<div style="width: 100%; height: 15px; border: 1px black solid"><svg viewbox="0 0 1 1" style="width:100%; height:100%" preserveAspectRatio="none">'
		body = '\n'.join('<rect data-begin="{begin}" data-end="{end}" x="{x}" width="{width}" height="1" style="fill:{color}" onclick="onclick_svg(event)"><title>speaker{speaker} | {begin:.2f} - {end:.2f} [{duration:.2f}]</title></rect>'.format(x = (t['begin'] - summary['begin']) / duration, width = (t['end'] - t['begin']) / duration, color = channel_colors[t['speaker']], duration = transcripts.compute_duration(t), **t) for t in transcript) 
		footer = '</svg></div>'
		html += header + body + footer
	return html

def audio_data_uri(audio_path, sample_rate = None, audio_backend = 'scipy', audio_format = 'wav'):
	data_uri = lambda audio_format, audio_bytes: f'data:audio/{audio_format};base64,' + base64.b64encode(audio_bytes).decode()
	
	if isinstance(audio_path, str):
		assert audio_path.endswith('.wav')
		audio_bytes, audio_format = open(audio_path, 'rb').read(), 'wav'
	else:
		audio_bytes = audio.write_audio(io.BytesIO(), audio_path, sample_rate, backend = audio_backend, format = audio_format).getvalue()
		
	return data_uri(audio_format = audio_format, audio_bytes = audio_bytes)


def label(output_path, transcript, info, page_size, prefix):
	if isinstance(transcript, str):
		transcript = json.load(open(transcript))
	if isinstance(info, str):
		info = json.load(open(info))
	transcript = {transcripts.audio_name(t): t for t in transcript}

	page_count = int(math.ceil(len(info) / page_size))
	for p in range(page_count):
		html_path = output_path + f'.page{p}.html'
		html = open(html_path, 'w')
		html.write(
			'<html><head><meta charset="UTF-8"><style>figure{margin:0} h6{margin:0}</style></head><body onkeydown="return onkeydown_(event)">'
		)
		html.write(
			'''<script>
			function export_user_input()
			{
				const data_text_plain_base64_encode_utf8 = str => 'data:text/plain;base64,' + btoa(encodeURIComponent(str).replace(/%([0-9A-F]{2})/g, function(match, p1) {return String.fromCharCode(parseInt(p1, 16)) }));
				
				const after = Array.from(document.querySelectorAll('input.after'));
				const data = after.map(input => ({audio_name : input.name, before : input.dataset.before, after : input.value}));

				const href = data_text_plain_base64_encode_utf8(JSON.stringify(data, null, 2));
				const unixtime = Math.round((new Date()).getTime() / 1000);
				let a = document.querySelector('a');
				const {page, prefix} = a.dataset;
				a.download = `${prefix}_page${page}_time${unixtime}.json`;
				a.href = href;
			}

			function onkeydown_(evt)
			{
					const tab = evt.keyCode == 9, shift = evt.shiftKey;
					const tabIndex = (document.activeElement || {tabIndex : -1}).tabIndex;
					if(tab)
					{
							const newTabIndex = shift ? Math.max(0, tabIndex - 1) : tabIndex + 1;
							const newElem = document.querySelector(`[tabindex="${newTabIndex}"`);
							if(newElem)
									newElem.focus();
							return false;
					}
					return true;
			}
		</script>'''
		)
		html.write(
			f'<a data-page="{p}" data-prefix="{prefix}" download="export.json" onclick="export_user_input(); return true" href="#">Export</a>\n'
		)

		k = p * page_size
		for j, i in enumerate(info[k:k + page_size]):
			i['after'] = i.get('after', '')
			t = transcript[i['audio_name']]
			html.write('<hr/>\n')
			html.write(
				f'<figure><figcaption>page {p}/{page_count}:<strong>{k + j}</strong><pre>{transcripts.audio_name(t)}</pre></figcaption><audio style="width:100%" controls src="{audio_data_uri(t["audio_path"][len("/data/"):])}"></audio><figcaption><pre>{t["ref"]}</pre></figcaption></figure>'
			)
			html.write('<h6>before</h6>')
			html.write('<pre name="{audio_name}" class="before">{before}</pre>'.format(**i))
			html.write('<h6>after</h6>')
			html.write(
				'<input tabindex="{tabindex}" name="{audio_name}" class="after" type="text" value="{after}" data-before="{before}">'
				.format(tabindex = j, **i)
			)
		html.write('</body></html>')
		print(html_path)


def transcript(html_path, sample_rate, mono, transcript, filtered_transcript = [], duration = None):
	if isinstance(transcript, str):
		transcript = json.load(open(transcript))

	has_hyp = any(t.get('hyp') for t in transcript)
	has_ref = any(t.get('ref') for t in transcript)

	audio_path = transcript[0]['audio_path']
	signal, sample_rate = audio.read_audio(audio_path, sample_rate = sample_rate, mono = mono, duration = duration)

	fmt_link = lambda ref = '', hyp = '', channel = 0, begin = 0, end = 0, speaker = '', i = '', j = '', audio_path = '', **kwargs: (f'<a onclick="return play({channel},{begin},{end})"' if ref not in [0, 1] else '<span') + f' title="#{channel}. {speaker}: {begin:.04f} - {end:.04f} | {i} - {j}" href="#" target="_blank">' + ((ref + hyp) if isinstance(ref, str) else f'{begin:.02f}' if ref == 0 else f'{end:.02f}' if ref == 1 else f'{end - begin:.02f}') + ('</a>' if ref not in [0, 1] else '</span>')
	fmt_words = lambda rh: ' '.join(fmt_link(**w) for w in rh)
	fmt_begin_end = 'data-begin="{begin}" data-end="{end}"'.format

	html = open(html_path, 'w')
	style = ' '.join(f'.speaker{i} {{background-color : {c}; }}' for i, c in enumerate(channel_colors)) + ' a {text-decoration: none;} .reference{opacity:0.4} .channel{margin:0px} .channel0 .hyp{padding-right:150px} .channel1 .hyp{padding-left:150px} .ok{background-color:green} .m0{margin:0px} .top{vertical-align:top} .channel0{background-color:violet} .channel1{background-color:lightblue; ' + ('display:none' if len(signal) == 1 else '') + '}'
	
	html.write(f'<html><head><meta charset="UTF-8"><style>{style}</style></head><body>')
	html.write(
		f'<div style="overflow:auto"><h4 style="float:left">{os.path.basename(audio_path)}</h4><h5 style="float:right">0.000000</h5></div>'
	)
	html_speaker_barcode = speaker_barcode_svg(transcript, begin = 0.0, end = signal.shape[-1] / sample_rate)

	html.writelines(
		f'<figure class="m0"><figcaption>channel #{c}:</figcaption><audio ontimeupdate="ontimeupdate_(event)" onpause="onpause_(event)" id="audio{c}" style="width:100%" controls src="{uri_audio}"></audio>{html_speaker_barcode}</figure>'
		for c,
		uri_audio in enumerate(
			audio_data_uri(signal[channel], sample_rate) for channel in ([0, 1] if len(signal) == 2 else []) + [...]
		)
	)
	html.write(
		f'<pre class="channel"><h3 class="channel0 channel">hyp #0:<span class="subtitle"></span></h3></pre><pre class="channel"><h3 class="channel0 reference channel">ref #0:<span class="subtitle"></span></h3></pre><pre class="channel" style="margin-top: 10px"><h3 class="channel1 channel">hyp #1:<span class="subtitle"></span></h3></pre><pre class="channel"><h3 class="channel1 reference channel">ref #1:<span class="subtitle"></span></h3></pre><hr/><table style="width:100%">'
	)
	def format_th(has_hyp, has_ref):
		speaker_th = '<th>speaker</th>'
		begin_th = '<th>begin</th>'
		end_th = '<th>end</th>'
		duration_th = '<th>dur</th>'
		hyp_th = '<th style="width:50%">hyp</th>' if has_hyp else ''
		ref_th = '<th style="width:50%">ref</th>' + begin_th + end_th + duration_th + '<th>cer</th>' if has_ref else ''
		return '<tr>' + speaker_th + begin_th + end_th + duration_th + hyp_th + ref_th

	html.write(format_th(has_hyp, has_ref))

	def format_tr(t, ok, has_hyp, has_ref, hyp, ref, channel, speaker):
		speaker_td = f'''<td class="top {ok and 'ok'} speaker{speaker}">{speaker}</td>'''
		begin_td = f'<td class="top">{fmt_link(0, **transcripts.summary(hyp, ij = True))}</td>'
		end_td = f'<td class="top">{fmt_link(1, **transcripts.summary(hyp, ij = True))}</td>'
		duration_td = f'<td class="top">{fmt_link(2, **transcripts.summary(hyp, ij = True))}</td>'
		hyp_td = '<td class="top hyp" data-channel="{c}" {fmt_begin_end(**transcripts.summary(hyp, ij = True))}>{fmt_words(hyp)}<template>{word_alignment(t.get("words", []), tag = "", hyp = True)}</template></td>' if has_hyp else ''
		ref_td = f'<td class="top reference ref" data-channel="{channel}" {fmt_begin_end(**transcripts.summary(ref, ij = True))}>{fmt_words(ref)}<template>{word_alignment(t.get("words", []), tag = "", ref = True)}</template></td><td class="top">{fmt_link(0, **transcripts.summary(ref, ij = True))}</td><td class="top">{fmt_link(1, **transcripts.summary(ref, ij = True))}</td><td class="top">{fmt_link(2, **transcripts.summary(ref, ij = True))}</td><td class="top">{t.get("cer", -1):.2%}</td>' if (has_ref and ref) else ('<td></td>' * 5 if has_ref else '')
		return f'<tr class="channel{channel}">' + speaker_td + begin_td + end_td + duration_td + hyp_td + ref_td + '</tr>'

	html.writelines(format_tr(t, ok, has_hyp, has_ref, hyp, ref, channel, speaker) for t in transcripts.sort(transcript)
		for ok in [t in filtered_transcript] for channel,
		speaker,
		ref,
		hyp in [(t.get('channel', 0), t.get('speaker', 0), t.get('words_ref', [t]), t.get('words_hyp', [t]))]
	)
	html.write(
		'''</tbody></table><script>
		function play(channel, begin, end, relative)
		{
			Array.from(document.querySelectorAll('audio')).map(audio => audio.pause());
			const audio = document.querySelector(`#audio${channel}`);
			if(relative)
				[begin, end] = [begin * audio.duration, end * audio.duration];
			audio.currentTime = begin;
			audio.dataset.endTime = end;
			audio.play();
			return false;
		}
		
		function onclick_img(evt)
		{
			const img = evt.target;
			const dim = img.getBoundingClientRect();
			const t = (evt.clientX - dim.left) / dim.width;
			play(0, t, 0 * audio.duration, true);
			audio.play();
		}
		
		function onclick_svg(evt)
		{
			const rect = evt.target;
			play(0, parseFloat(rect.dataset.begin), parseFloat(rect.dataset.end));
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
			if(endtime && endtime > 0 && time > endtime)
				return evt.target.pause();

			document.querySelector('h5').innerText = time.toString();
			const [spanhyp0, spanref0, spanhyp1, spanref1] = document.querySelectorAll('span.subtitle');
			[spanhyp0.innerHTML, spanref0.innerHTML, spanhyp1.innerHTML, spanref1.innerHTML] = [subtitle(hyp_segments, time, 0), subtitle(ref_segments, time, 0), subtitle(hyp_segments, time, 1), subtitle(ref_segments, time, 1)];
		}

		const make_segment = td => [td.querySelector('template').innerHTML, td.dataset.channel, td.dataset.begin, td.dataset.end];
		const hyp_segments = Array.from(document.querySelectorAll('.hyp')).map(make_segment), ref_segments = Array.from(document.querySelectorAll('.ref')).map(make_segment);
	</script></body></html>'''
	)
	print(html_path)


def logits(logits, audio_name, MAX_ENTROPY = 1.0):
	good_audio_name = set(map(str.strip, open(audio_name[0])) if os.path.exists(audio_name[0]) else audio_name)
	labels = datasets.Labels(ru)
	decoder = decoders.GreedyDecoder()
	tick_params = lambda ax, labelsize = 2.5, length = 0, **kwargs: ax.tick_params(axis = 'both', which = 'both', labelsize = labelsize, length = length, **kwargs) or [ax.set_linewidth(0) for ax in ax.spines.values()]
	logits_path = logits + '.html'
	html = open(logits_path, 'w')
	html.write(
		'''<html><meta charset="utf-8"/><body><script>
		function onclick_(evt)
		{
			const img = evt.target;
			const dim = img.getBoundingClientRect();
			const t = (evt.clientX - dim.left) / dim.width;
			const audio = img.nextSibling;
			audio.currentTime = t * audio.duration;
			audio.play();
		}
	</script>'''
	)
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

		alignment = ctc.alignment(
			log_probs.unsqueeze(0).permute(2, 0, 1),
			r['y'].unsqueeze(0).long(),
			torch.LongTensor([log_probs.shape[-1]]),
			torch.LongTensor([len(r['y'])]),
			blank = len(log_probs) - 1
		).squeeze(0)

		plt.figure(figsize = (6, 2))

		prob_top1, prob_top2 = log_probs.exp().topk(2, dim = 0).values
		plt.hlines(1.0, 0, entropy.shape[-1] - 1, linewidth = 0.2)
		artist_prob_top1, = plt.plot(prob_top1, 'b', linewidth = 0.3)
		artist_prob_top2, = plt.plot(prob_top2, 'g', linewidth = 0.3)
		artist_entropy, = plt.plot(entropy, 'r', linewidth = 0.3)
		artist_entropy_, = plt.plot(entropy_, 'yellow', linewidth = 0.3)
		plt.legend([artist_entropy, artist_entropy_, artist_prob_top1, artist_prob_top2],
					['entropy', 'entropy, no blank', 'top1 prob', 'top2 prob'],
					loc = 1,
					fontsize = 'xx-small',
					frameon = False)
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
		xlabels = list(
			map(
				'\n'.join,
				zip(
					*[
						labels.decode(d, replace_blank = '.', replace_space = '_', replace_repeat = False)
						for d in decoded
					]
				)
			)
		)
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
		html.write(
			'<img onclick="onclick_(event)" style="width:100%" src="data:image/jpeg;base64,{encoded}"></img>'.format(
				encoded = base64.b64encode(buf.getvalue()).decode()
			)
		)
		html.write('<audio style="width:100%" controls src="{audio_data_uri(r["audio_path"])}"></audio><hr/>')
	html.write('</body></html>')
	print('\n', logits_path)


def errors(
	input_path,
	include = [],
	exclude = [],
	audio = False,
	output_path = None,
	sortdesc = None,
	topk = None,
	duration = None,
	cer = None,
	wer = None,
	mer = None,
	filter_transcripts = None,
	strip_audio_path_prefix = ''
):
	include, exclude = (set(sum([list(map(transcripts.audio_name, json.load(open(file_path)))) if file_path.endswith('.json') else open(file_path).read().splitlines() for file_path in clude], [])) for clude in [include, exclude])
	read_transcript = lambda path: list(
		filter(
			lambda r: (not include or r['audio_name'] in include) and (not exclude or r['audio_name'] not in exclude),
			json.load(open(path)) if isinstance(path, str) else path
		)
	) if path is not None else []
	ours, theirs = list(transcripts.prune(read_transcript(input_path[0]), duration = duration, cer = cer, wer = wer, mer = mer)), [{r['audio_name'] : r for r in read_transcript(transcript)} for transcript in input_path[1:]]
	
	if filter_transcripts is None:
		if sortdesc is not None:
			filter_transcripts = lambda cat: list(sorted(cat, key = lambda utt: utt[0][sortdesc], reverse = True))
		else:
			filter_transcripts = lambda cat: cat

	cat = filter_transcripts([[a] + list(filter(None, [t.get(a['audio_name'], None)
														for t in theirs]))
								for a in ours])[slice(topk)]
	cat_by_labels = collections.defaultdict(list)
	for c in cat:
		transcripts_by_labels = collections.defaultdict(list)
		for transcript in c:
			transcripts_by_labels[transcript['labels_name']] += c
		for labels_name, grouped_transcripts in transcripts_by_labels.items():
			cat_by_labels[labels_name] += grouped_transcripts

	# TODO: add sorting https://stackoverflow.com/questions/14267781/sorting-html-table-with-javascript
	html_path = output_path or (input_path[0] + '.html')

	f = open(html_path, 'w')
	f.write(
		'<html><meta charset="utf-8"><style> table{border-collapse:collapse; width: 100%;} audio {width:100%} .br{border-right:2px black solid} tr.first>td {border-top: 1px solid black} tr.any>td {border-top: 1px dashed black}  .nowrap{white-space:nowrap} th.col{width:80px}</style>'
	)
	f.write(
		'<body><table><tr><th></th><th class="col">cer_easy</th><th class="col">cer</th><th class="col">wer_easy</th><th class="col">wer</th><th class="col">mer</th><th></th></tr>'
	)
	f.write('<tr><td><strong>averages<strong></td></tr>')
	f.write(
		'\n'.join(
			'<tr><td class="br">{input_name}</td><td>{cer_easy:.02%}</td><td>{cer:.02%}</td><td>{wer_easy:.02%}</td><td>{wer:.02%}</td><td>{mer:.02%}</td></tr>'
			.format(
				input_name = os.path.basename(input_path[i]),
				cer = metrics.nanmean(c, 'cer'),
				wer = metrics.nanmean(c, 'wer'),
				mer = metrics.nanmean(c, 'mer'),
				cer_easy = metrics.nanmean(c, 'words_easy_errors_easy.cer_pseudo'),
				wer_easy = metrics.nanmean(c, 'words_easy_errors_easy.wer_pseudo'),
			) for i,
			c in enumerate(zip(*cat))
		)
	)
	if len(cat_by_labels.keys()) > 1:
		for labels_name, labels_transcripts in cat_by_labels.items():
			f.write(f'<tr><td><strong>averages ({labels_name})<strong></td></tr>')
			f.write(
				'\n'.join(
					'<tr><td class="br">{input_name}</td><td>{cer_easy:.02%}</td><td>{cer:.02%}</td><td>{wer_easy:.02%}</td><td>{wer:.02%}</td><td>{mer:.02%}</td></tr>'
					.format(
						input_name = os.path.basename(input_path[i]),
						cer = metrics.nanmean(c, 'cer'),
						wer = metrics.nanmean(c, 'wer'),
						mer = metrics.nanmean(c, 'mer_wordwise'),
						cer_easy = metrics.nanmean(c, 'words_easy_errors_easy.cer_pseudo'),
						wer_easy = metrics.nanmean(c, 'words_easy_errors_easy.wer_pseudo'),
					) for i,
					c in enumerate(zip(*labels_transcripts))
				)
			)
	f.write('<tr><td>&nbsp;</td></tr>')
	f.write(
		'\n'.join(
			f'''<tr class="first"><td colspan="6">''' + (
				f'<audio controls src="{audio_data_uri(utt[0]["audio_path"][len(strip_audio_path_prefix):])}"></audio>'
				if audio else ''
			) +
			f'<div class="nowrap">{utt[0]["audio_name"]}</div></td><td>{word_alignment(utt[0], ref = True, flat = True)}</td><td>{word_alignment(utt[0], ref = True, flat = True)}</td></tr>'
			+ '\n'.join(
				'<tr class="any"><td class="br">{audio_name}</td><td>{cer_easy:.02%}</td><td>{cer:.02%}</td><td>{wer_easy:.02%}</td><td>{wer:.02%}</td><td class="br">{mer:.02%}</td><td>{word_alignment}</td><td>{word_alignment_flat}</td></tr>'.format(audio_name = transcripts.audio_name(input_path[i]), cer_easy = a.get("words_easy_errors_easy", {}).get("cer_pseudo", -1), cer = a.get("cer", 1), wer_easy = a.get("words_easy_errors_easy", {}).get("wer_pseudo", -1), wer = a.get("wer", 1), mer = a.get("mer_wordwise", 1), word_alignment = word_alignment(a.get('words', [])), word_alignment_flat = word_alignment(a, hyp = True, flat = True))
				for i,
				a in enumerate(utt)
			)
			for utt in cat
		)
	)
	f.write('</table></body></html>')
	print(html_path)


def audiosample(input_path, output_path, K):
	transcript = json.load(open(input_path))

	group = lambda t: t.get('group', 'group not found')
	by_group = {k: list(g) for k, g in itertools.groupby(sorted(transcript, key = group), key = group)}

	f = open(output_path, 'w')
	f.write('<html><meta charset="UTF-8"><body>')
	for group, transcript in sorted(by_group.items()):
		f.write(f'<h1>{group}</h1>')
		f.write('<table>')
		random.seed(1)
		random.shuffle(transcript)
		for t in transcript[:K]:
			try:
				data_uri = audio_data_uri(os.path.join(args.dataset_root, t['audio_path']))
			except:
				f.write('<tr><td>file not found: {audio_path}</td></tr>'.format(**t))
				continue
			f.write(
				'<tr><td>{audio_path}</td><td><audio controls src="{data_uri}"/></td><td>{ref}</td></tr>\n'.format(
					encoded = encoded, **t
				)
			)
		f.write('</table>')

	print(output_path)


def summary(input_path, lang):
	lang = datasets.Language(lang)
	transcript = json.load(open(input_path))
	for t in transcript:
		hyp, ref = map(lang.normalize_text, [t['hyp'], t['ref']])
		t['cer'] = t.get('cer', metrics.cer(hyp, ref))
		t['wer'] = t.get('wer', metrics.wer(hyp, ref))

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


def tabulate(
	experiments_dir,
	experiment_id,
	entropy,
	loss,
	cer10,
	cer15,
	cer20,
	cer30,
	cer40,
	cer50,
	per,
	wer,
	json_,
	bpe,
	der,
	lang
):
	# TODO: bring back custom name to the filtration process, or remove filtration by labels_name entirely.
	labels = datasets.Labels(lang=datasets.Language(lang), name='char')

	res = collections.defaultdict(list)
	experiment_dir = os.path.join(experiments_dir, experiment_id)
	for f in sorted(glob.glob(os.path.join(experiment_dir, f'transcripts_*.json'))):
		eidx = f.find('epoch')
		iteration = f[eidx:].replace('.json', '')
		val_dataset_name = f[f.find('transcripts_') + len('transcripts_'):eidx]
		checkpoint = os.path.join(experiment_dir, 'checkpoint_' + f[eidx:].replace('.json', '.pt')) if not json_ else f
		metric = 'wer' if wer else 'entropy' if entropy else 'loss' if loss else 'per' if per else 'der' if der else 'cer'
		val = torch.tensor([j[metric] for j in json.load(open(f)) if j['labels_name'] == labels.name] or [0.0])
		val = val[~(torch.isnan(val) | torch.isinf(val))]

		if cer10 or cer20 or cer30 or cer40 or cer50:
			val = (val < 0.1 * [False, cer10, cer20, cer30, cer40, cer50].index(True)).float()
		if cer15:
			val = (val < 0.15).float()
		res[iteration].append((val_dataset_name, float(val.mean()), checkpoint))
	val_dataset_names = sorted(set(val_dataset_name for r in res.values() for val_dataset_name, cer, checkpoint in r))
	print('iteration\t' + '\t'.join(val_dataset_names))
	for iteration, r in res.items():
		cers = {val_dataset_name: f'{cer:.04f}' for val_dataset_name, cer, checkpoint in r}
		print(
			f'{iteration}\t' + '\t'.join(cers.get(val_dataset_name, '')
											for val_dataset_name in val_dataset_names) + f'\t{r[-1][-1]}'
		)


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
	return altair.Chart(altair.Data(values = [dict(x = b, y = v) for b, v in zip(bins.tolist(), hist.tolist())])
						).mark_bar().encode(x = altair.X('x:Q'), y = altair.Y('y:Q')).to_dict()


def word_alignment(transcript, ref = None, hyp = None, flat = False, tag = '<pre>', prefix = True):
	span = lambda word, t = None: '<span style="{style}" title="{word_alignment_error_type}">{word}</span>'.format(word = word, style = ('background-color:' + dict(ok = 'green', missing = 'red', missing_ref = 'darkred', typo_easy = 'lightgreen', typo_hard = 'pink')[t]) if t is not None else '', word_alignment_error_type = t)

	error_tag = lambda w: w.get('type') or w.get('error_tag')
	if flat:
		ref_ = transcript.get('ref', '')
		hyp_ = transcript.get('hyp', '')
	else:
		ref_ = ' '.join(span(w['ref'], 'ok' if error_tag(w) == 'ok' else None) for w in transcript)
		hyp_ = ' '.join(span(w['hyp'], error_tag(w)) for w in transcript)

	ref_ = ('ref: ' if prefix else '') + ref_
	hyp_ = ('hyp: ' if prefix else '') + hyp_
	contents = '\n'.join([ref_] if ref is True else [hyp_] if hyp is True else [ref_, hyp_])
	return tag + contents + tag.replace('<', '</')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()

	cmd = subparsers.add_parser('label')
	cmd.add_argument('--transcript', '-i')
	cmd.add_argument('--info')
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--page-size', type = int, default = 100)
	cmd.add_argument('--prefix', default = 'export')
	cmd.set_defaults(func = label)

	cmd = subparsers.add_parser('transcript')
	cmd.add_argument('--transcript', '-i')
	cmd.add_argument('--mono', action = 'store_true')
	cmd.add_argument('--sample-rate', type = int, default = 8_000)
	cmd.add_argument('--html-path', '-o')
	cmd.set_defaults(func = transcript)

	cmd = subparsers.add_parser('errors')
	cmd.add_argument('input_path', nargs = '+', default = ['data/transcripts.json'])
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--include', nargs = '*', default = [])
	cmd.add_argument('--exclude', nargs = '*', default = [])
	cmd.add_argument('--audio', action = 'store_true')
	cmd.add_argument('--sortdesc', choices = ['cer', 'wer', 'mer'])
	cmd.add_argument('--topk', type = int)
	cmd.add_argument('--cer', type = transcripts.number_tuple)
	cmd.add_argument('--wer', type = transcripts.number_tuple)
	cmd.add_argument('--mer', type = transcripts.number_tuple)
	cmd.add_argument('--duration', type = transcripts.number_tuple)
	cmd.add_argument('--strip-audio-path-prefix', default = '')
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
	cmd.add_argument('--lang', default = 'ru')
	cmd.set_defaults(func = tabulate)

	cmd = subparsers.add_parser('summary')
	cmd.add_argument('input_path')
	cmd.add_argument('--lang', default = 'ru')
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
