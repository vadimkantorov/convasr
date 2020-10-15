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

meta_charset = '<meta charset="UTF-8">'

onclick_img_script = '''
function onclick_img(evt)
{
	const img = evt.target;
	const dim = img.getBoundingClientRect();
	let begin = (evt.clientX - dim.left) / dim.width;
	let relative = true;
	if(img.dataset.begin != null && img.dataset.begin != '' && img.dataset.end != null && img.dataset.end != '')
	{
		begin = parseFloat(img.dataset.begin) + (parseFloat(img.dataset.end) - parseFloat(img.dataset.begin)) * begin;
		relative = false;
	}
	const channel = img.dataset.channel || 0;
	play(evt, channel, begin, 0, false);
}
'''

onclick_svg_script = '''
function onclick_svg(evt)
{
	const rect = evt.target;
	const channel = rect.dataset.channel || 0;
	play(evt, channel, parseFloat(rect.dataset.begin), parseFloat(rect.dataset.end));
}
'''

play_script = '''
var playTimeStampMillis = 0.0;

function download_audio(evt, channel)
{
	const a = evt.target;
	a.href = document.getElementById(`audio${channel}`).src;
	return true;
}

function play(evt, channel, begin, end, relative)
{
	Array.from(document.querySelectorAll('audio')).map(audio => audio.pause());
	const audio = document.querySelector(`#audio${channel}`);
	if(relative)
		[begin, end] = [begin * audio.duration, end * audio.duration];
	audio.currentTime = begin;
	audio.dataset.endTime = end;
	playTimeStampMillis = evt.timeStamp;
	audio.play();
	return false;
}

function onpause_(evt)
{
	if(evt.timeStamp - playTimeStampMillis > 10)
		evt.target.dataset.endTime = null;
}

function ontimeupdate_(evt)
{
	const time = evt.target.currentTime, endtime = evt.target.dataset.endTime;
	if(endtime && endtime > 0 && time > endtime)
	{
		evt.target.pause();
		return false;
	}
	return true;
}
'''

subtitle_script = '''
function subtitle(segments, time, channel, speaker)
{
	return (segments.find(([rh, c, s, b, e]) => (c == channel || s == speaker) && b <= time && time <= e ) || ['', channel, speaker, null, null])[0];
}

function update_span(proceed, evt)
{
	if(!proceed)
		return false;

	const time = evt.target.currentTime;
	document.querySelector('h5').innerText = time.toString();
	const [spanhyp0, spanref0, spanhyp1, spanref1] = document.querySelectorAll('span.subtitle');
	[spanhyp0.innerHTML, spanref0.innerHTML, spanhyp1.innerHTML, spanref1.innerHTML] = [subtitle(hyp_segments, time, 0, 1), subtitle(ref_segments, time, 0, 1), subtitle(hyp_segments, time, 1, 2), subtitle(ref_segments, time, 1, 2)];
}

const make_segment = td => [td.querySelector('template').innerHTML, td.dataset.channel, td.dataset.speaker, td.dataset.begin, td.dataset.end];
const hyp_segments = Array.from(document.querySelectorAll('.hyp')).map(make_segment);
const ref_segments = Array.from(document.querySelectorAll('.ref')).map(make_segment);
'''

channel_colors = ['violet', 'lightblue']
speaker_colors = ['gray', 'violet', 'lightblue']
#speaker_colors = ['gray', 'red', 'blue']

def diarization(diarization_transcript, html_path, debug_audio):
	with open(html_path, 'w') as html:
		html.write('<html><head>' + meta_charset + '<style>.nowrap{white-space:nowrap} table {border-collapse:collapse} .border-hyp {border-bottom: 2px black solid}</style></head><body>\n')
		html.write(f'<script>{play_script}</script>\n')
		html.write(f'<script>{onclick_img_script}</script>')
		html.write('<table>\n')
		html.write('<tr><th>audio_name</th><th>duration</th><th>refhyp</th><th>ser</th><th>der</th><th>der_</th><th>audio</th><th>barcode</th></tr>\n')
		avg = lambda l: sum(l) / len(l)
		html.write('<tr class="border-hyp"><td>{num_files}</td><td>{total_duration:.02f}</td><td>avg</td><td>{avg_ser:.02f}</td><td>{avg_der:.02f}</td><td>{avg_der_:.02f}</td><td></td><td></td></tr>\n'.format(
			num_files = len(diarization_transcript), 
			total_duration = sum(map(transcripts.compute_duration, diarization_transcript)), 
			avg_ser = avg([t['ser'] for t in diarization_transcript]),
			avg_der = avg([t['der'] for t in diarization_transcript]),
			avg_der_ = avg([t['der_'] for t in diarization_transcript])
		))
		for i, dt in enumerate(diarization_transcript):
			audio_html = fmt_audio(audio_path, channel = channel) if debug_audio else ''
			begin, end = 0.0, transcripts.compute_duration(dt)
			for refhyp in ['ref', 'hyp']:
				html.write('<tr class="border-{refhyp}"><td class="nowrap">{audio_name}</td><td>{end:.02f}</td><td>{refhyp}</td><td>{ser:.02f}</td><td>{der:.02f}</td><td>{der_:.02f}</td><td rospan="{rowspan}">{audio_html}</td><td>{barcode}</td></tr>\n'.format(audio_name = dt['audio_name'], audio_html = audio_html if refhyp == 'ref' else '', rowspan = 2 if refhyp == 'ref' else 1, refhyp = refhyp, end = end, ser = dt['ser'], der = dt['der'], der_ = dt['der_'], barcode = fmt_img_speaker_barcode(dt[refhyp], begin = begin, end = end, onclick = None if debug_audio else '', dataset = dict(channel = i))))

		html.write('</table></body></html>')
	return html_path

def fmt_img_speaker_barcode(transcript, begin = None, end = None, colors = speaker_colors, onclick = None, dataset = {}):
	if begin is None:
		begin = 0
	if end is None:
		end = max(t['end'] for t in transcript)
	if onclick is None:
		onclick = 'onclick_img(event)'
	color = lambda s: colors[s] if s < len(colors) else transcripts.speaker_missing

	plt.figure(figsize = (8, 0.2))
	plt.xlim(begin, end)
	plt.yticks([])
	plt.axis('off')
	for t in transcript:
		plt.axvspan(t['begin'], t['end'], color = color(t.get('speaker', transcripts.speaker_missing)))
	plt.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
	buf = io.BytesIO()
	plt.savefig(buf, format = 'jpg', dpi = 150, facecolor = color(transcripts.speaker_missing))
	plt.close()
	uri_speaker_barcode = base64.b64encode(buf.getvalue()).decode()
	dataset = ' '.join(f'data-{k}="{v}"' for k, v in dataset.items())
	return f'<img onclick="{onclick}" src="data:image/jpeg;base64,{uri_speaker_barcode}" style="width:100% {dataset}"></img>'


def fmt_svg_speaker_barcode(transcript, begin, end, colors = speaker_colors, max_segment_seconds = 60, onclick = None):
	if onclick is None:
		onclick = 'onclick_svg(event)'
	color = lambda s: colors[s] if s < len(colors) else transcripts.speaker_missing
	html = ''
	
	segments = transcripts.segment_by_time(transcript, max_segment_seconds = max_segment_seconds, break_on_speaker_change = False, break_on_channel_change = False)
	
	for segment in segments:
		summary = transcripts.summary(segment)
		duration = transcripts.compute_duration(summary)
		if duration <= max_segment_seconds:
			duration = max_segment_seconds
		header = '<div style="width: 100%; height: 15px; border: 1px black solid"><svg viewbox="0 0 1 1" style="width:100%; height:100%" preserveAspectRatio="none">'
		body = '\n'.join('<rect data-begin="{begin}" data-end="{end}" x="{x}" width="{width}" height="1" style="fill:{color}" onclick="{onclick}"><title>speaker{speaker} | {begin:.2f} - {end:.2f} [{duration:.2f}]</title></rect>'.format(onclick = onclick, x = (t['begin'] - summary['begin']) / duration, width = (t['end'] - t['begin']) / duration, color = color(t['speaker']), duration = transcripts.compute_duration(t), **t) for t in transcript) 
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

def fmt_audio(audio_path, channel = 0):
	return f'<audio id="audio{channel}" style="width:100%" controls src="{audio_data_uri(audio_path)}"></audio>\n'

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
			'<html><head>' + meta_charset + '<style>figure{margin:0} h6{margin:0}</style></head><body onkeydown="return onkeydown_(event)">'
		)
		html.write('''<script>
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
			audio_path = t['audio_path'][len('/data/'):]
			html.write('<hr/>\n')
			html.write(
				f'<figure><figcaption>page {p}/{page_count}:<strong>{k + j}</strong><pre>{transcripts.audio_name(t)}</pre></figcaption>{fmt_audio(audio_path)}<figcaption><pre>{t["ref"]}</pre></figcaption></figure>'
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


def transcript(html_path, sample_rate, mono, transcript, filtered_transcript = [], duration = None, NA = 'N/A', default_channel = 0):
	if isinstance(transcript, str):
		transcript = json.load(open(transcript))

	audio_path = transcript[0]['audio_path']
	audio_name = transcripts.audio_name(audio_path)

	signal, sample_rate = audio.read_audio(audio_path, sample_rate = sample_rate, mono = mono, duration = duration)
	channel_or_default = lambda channel: default_channel if channel == transcripts.channel_missing else channel 

	def fmt_link(ref = '', hyp = '', channel = default_channel, begin = transcripts.time_missing, end = transcripts.time_missing, speaker = transcripts.speaker_missing, i = '', j = '', audio_path = '', special_begin = 0, special_end = 1, **kwargs):
		span = ref in [special_begin, special_end] or begin == transcripts.time_missing or end == transcripts.time_missing
		tag_open = '<span' if span else f'<a onclick="return play(event, {channel_or_default(channel)}, {begin}, {end})"'
		tag_attr = f' title="channel{channel}. speaker{speaker}: {begin:.04f} - {end:.04f} | {i} - {j}" href="#" target="_blank">'
		tag_contents = (ref + hyp) if isinstance(ref, str) else (f'{begin:.02f}' if begin != transcripts.time_missing else NA) if ref == special_begin else (f'{end:.02f}' if end != transcripts.time_missing else NA) if ref == special_end else (f'{end - begin:.02f}' if begin != transcripts.time_missing and end != transcripts.time_missing else NA)
		tag_close = '</span>' if span else '</a>'
		return tag_open + tag_attr + tag_contents + tag_close
	
	fmt_words = lambda rh: ' '.join(fmt_link(**w) for w in rh)
	fmt_begin_end = 'data-begin="{begin}" data-end="{end}"'.format

	html = open(html_path, 'w')
	style = ' '.join(f'.speaker{i} {{background-color : {c}; }}' for i, c in enumerate(speaker_colors)) + ' '.join(f'.channel{i} {{background-color : {c}; }}' for i, c in enumerate(channel_colors)) + ' a {text-decoration: none;} .reference{opacity:0.4} .channel{margin:0px} .ok{background-color:green} .m0{margin:0px} .top{vertical-align:top}' 
	
	html.write(f'<html><head>' + meta_charset + f'<style>{style}</style></head><body>')
	html.write(f'<script>{play_script}{onclick_svg_script}</script>')
	html.write(
		f'<div style="overflow:auto"><h4 style="float:left">{audio_name}</h4><h5 style="float:right">0.000000</h5></div>'
	)
	html_speaker_barcode = fmt_svg_speaker_barcode(transcript, begin = 0.0, end = signal.shape[-1] / sample_rate)

	html.writelines(
		f'<figure class="m0"><figcaption><a href="#" download="channel{c}.{audio_name}" onclick="return download_audio(event, {c})">channel #{c}:</a></figcaption><audio ontimeupdate="update_span(ontimeupdate_(event), event)" onpause="onpause_(event)" id="audio{c}" style="width:100%" controls src="{uri_audio}"></audio>{html_speaker_barcode}</figure><hr />'
		for c,
		uri_audio in enumerate(
			audio_data_uri(signal[channel], sample_rate) for channel in ([0, 1] if len(signal) == 2 else []) + [...]
		)
	)
	html.write('<pre class="channel"><h3 class="channel0 channel">hyp #0:<span class="subtitle"></span></h3></pre>')
	html.write('<pre class="channel"><h3 class="channel0 reference channel">ref #0:<span class="subtitle"></span></h3></pre>')
	html.write('<pre class="channel" style="margin-top: 10px"><h3 class="channel1 channel">hyp #1:<span class="subtitle"></span></h3></pre>')
	html.write('<pre class="channel"><h3 class="channel1 reference channel">ref #1:<span class="subtitle"></span></h3></pre>')

	def fmt_th():
		idx_th = '<th>#</th>'
		speaker_th = '<th>speaker</th>'
		begin_th = '<th>begin</th>'
		end_th = '<th>end</th>'
		duration_th = '<th>dur</th>'
		hyp_th = '<th style="width:50%">hyp</th>'
		ref_th = '<th style="width:50%">ref</th>' + begin_th + end_th + duration_th + '<th>cer</th>' 
		return '<tr>' + idx_th + speaker_th + begin_th + end_th + duration_th + hyp_th + ref_th
 
	def fmt_tr(i, ok, t, words, hyp, ref, channel, speaker, speaker_name, cer):
		idx_td = f'''<td class="top {ok and 'ok'}">#{i}</td>'''
		speaker_td = f'<td class="speaker{speaker}" title="speaker{speaker}">{speaker_name}</td>'
		left_td = f'<td class="top">{fmt_link(0, **transcripts.summary(hyp, ij = True))}</td><td class="top">{fmt_link(1, **transcripts.summary(hyp, ij = True))}</td><td class="top">{fmt_link(2, **transcripts.summary(hyp, ij = True))}</td>'
		hyp_td = f'<td class="top hyp" data-channel="{channel}" data-speaker="{speaker}" {fmt_begin_end(**transcripts.summary(hyp, ij = True))}>{fmt_words(hyp)}{fmt_alignment(words, hyp = True, prefix = "", tag = "<template>")}</td>'
		ref_td = f'<td class="top reference ref" data-channel="{channel}" data-speaker="{speaker}" {fmt_begin_end(**transcripts.summary(ref, ij = True))}>{fmt_words(ref)}{fmt_alignment(words, ref = True, prefix = "", tag = "<template>")}</td>'
		right_td = f'<td class="top">{fmt_link(0, **transcripts.summary(ref, ij = True))}</td><td class="top">{fmt_link(1, **transcripts.summary(ref, ij = True))}</td><td class="top">{fmt_link(2, **transcripts.summary(ref, ij = True))}</td>'
		cer_td = f'<td class="top">' + (f'{cer:.2%}' if cer != transcripts._er_missing else NA) + '</td>' 
		return f'<tr class="channel{channel} speaker{speaker}">' + idx_td + speaker_td + left_td + hyp_td + ref_td + right_td + cer_td + '</tr>\n'

	html.write('<hr/><table style="width:100%">')
	html.write(fmt_th())
	html.writelines(fmt_tr(i, t in filtered_transcript, t, t.get('words', [t]), t.get('words_hyp', [t]), t.get('words_ref', [t]), t.get('channel', transcripts.channel_missing), t.get('speaker', transcripts.speaker_missing), t.get('speaker_name', 'speaker{}'.format(t.get('speaker', transcripts.speaker_missing))), t.get('cer', transcripts._er_missing)) for i, t in enumerate(transcripts.sort(transcript)))
	html.write(f'</tbody></table><script>{subtitle_script}</script></body></html>')
	return html_path


def logits(lang, logits, audio_name = None, MAX_ENTROPY = 1.0):
	good_audio_name = set(map(str.strip, open(audio_name[0])) if os.path.exists(audio_name[0]) else audio_name) if audio_name is not None else []
	labels = datasets.Labels(datasets.Language(lang))
	decoder = decoders.GreedyDecoder()
	tick_params = lambda ax, labelsize = 2.5, length = 0, **kwargs: ax.tick_params(axis = 'both', which = 'both', labelsize = labelsize, length = length, **kwargs) or [ax.set_linewidth(0) for ax in ax.spines.values()]
	logits_path = logits + '.html'
	html = open(logits_path, 'w')
	html.write('<html><head>' + meta_charset + f'</head><body><script>{play_script}{onclick_img_script}</script>')

	for i, t in enumerate(torch.load(logits)):
		audio_path, logits = t['audio_path'], t['logits']
		words = t.get('words', [t])
		y = t.get('y', torch.zeros(1, 0, dtype = torch.long))
		begin = t.get('begin', '')
		end = t.get('end', '')
		audio_name = transcripts.audio_name(audio_path)
		extra_metrics = dict(cer = t['cer']) if 'cer' in t else {}

		if good_audio_name and audio_name not in good_audio_name:
			continue

		log_probs = F.log_softmax(logits, dim = 0)
		entropy = models.entropy(log_probs, dim = 0, sum = False)
		log_probs_ = F.log_softmax(logits[:-1], dim = 0)
		entropy_ = models.entropy(log_probs_, dim = 0, sum = False)
		margin = models.margin(log_probs, dim = 0)
		#energy = features.exp().sum(dim = 0)[::2]

		plt.figure(figsize = (6, 2))
		ax = plt.subplot(211)
		plt.imshow(logits, aspect = 'auto')
		plt.xlim(0, logits.shape[-1] - 1)
		#plt.yticks([])
		plt.axis('off')
		tick_params(plt.gca())
		#plt.subplots_adjust(left = 0, right = 1, bottom = 0.12, top = 0.95)

		plt.subplot(212, sharex = ax)
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
		
		for b, e, v in zip(*models.rle1d(entropy > MAX_ENTROPY)):
			if bool(v):
				plt.axvspan(int(b), int(e), color='red', alpha=0.2)

		plt.ylim(0, 3.0)
		plt.xlim(0, entropy.shape[-1] - 1)

		decoded = decoder.decode(log_probs.unsqueeze(0), K = 5)[0]
		xlabels = list(
			map(
				'\n'.join,
				zip(
					*[
						labels.decode(d, replace_blank = '.', replace_space = '_', replace_repeat = False, strip = False)
						for d in decoded
					]
				)
			)
		)
		plt.xticks(torch.arange(entropy.shape[-1]), xlabels, fontfamily = 'monospace')
		tick_params(plt.gca())

		if y.numel() > 0:
			alignment = ctc.alignment(
				log_probs.unsqueeze(0).permute(2, 0, 1),
				y.unsqueeze(0).long(),
				torch.LongTensor([log_probs.shape[-1]]),
				torch.LongTensor([len(y)]),
				blank = len(log_probs) - 1
			).squeeze(0)
			
			ax = plt.gca().secondary_xaxis('top')
			ref, ref_ = labels.decode(y.tolist(), replace_blank = '.', replace_space = '_', replace_repeat = False, strip = False), alignment
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

		html.write(f'<h4>{audio_name}')
		html.write(' | '.join('{k}: {v:.02f}' for k, v in extra_metrics.items()))
		html.write('</h4>')
		html.write(fmt_alignment(words))
		html.write('<img data-begin="{begin}" data-end="{end}" data-channel="{channel}" onclick="onclick_img(event)" style="width:100%" src="data:image/jpeg;base64,{encoded}"></img>\n'.format(channel = i, begin = begin, end = end, encoded = base64.b64encode(buf.getvalue()).decode()))
		html.write(fmt_audio(audio_path = audio_path, channel = i))
		html.write('<hr/>')
	html.write('</body></html>')
	return logits_path


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
	f.write('<html><head>' + meta_charset)
	f.write('<style> table{border-collapse:collapse; width: 100%;} audio {width:100%} .br{border-right:2px black solid} tr.first>td {border-top: 1px solid black} tr.any>td {border-top: 1px dashed black}  .nowrap{white-space:nowrap} th.col{width:80px}</style></head>')
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
			'<tr class="first"><td colspan="6">' + (
				fmt_audio(utt[0]["audio_path"][len(strip_audio_path_prefix):]) if audio else ''
			) +
			f'<div class="nowrap">{utt[0]["audio_name"]}</div></td><td>{fmt_alignment(utt[0], ref = True, flat = True)}</td><td>{fmt_alignment(utt[0], ref = True, flat = True)}</td></tr>'
			+ '\n'.join(
				'<tr class="any"><td class="br">{audio_name}</td><td>{cer_easy:.02%}</td><td>{cer:.02%}</td><td>{wer_easy:.02%}</td><td>{wer:.02%}</td><td class="br">{mer:.02%}</td><td>{fmt_alignment}</td><td>{fmt_alignment_flat}</td></tr>'.format(audio_name = transcripts.audio_name(input_path[i]), cer_easy = a.get("words_easy_errors_easy", {}).get("cer_pseudo", -1), cer = a.get("cer", 1), wer_easy = a.get("words_easy_errors_easy", {}).get("wer_pseudo", -1), wer = a.get("wer", 1), mer = a.get("mer_wordwise", 1), fmt_alignment = fmt_alignment(a.get('words', [])), fmt_alignment_flat = fmt_alignment(a, hyp = True, flat = True))
				for i,
				a in enumerate(utt)
			)
			for utt in cat
		)
	)
	f.write('</table></body></html>')
	return html_path


def audiosample(input_path, output_path, K):
	transcript = json.load(open(input_path))

	group = lambda t: t.get('group', 'group not found')
	by_group = {k: list(g) for k, g in itertools.groupby(sorted(transcript, key = group), key = group)}

	f = open(output_path, 'w')
	f.write(f'<html><head>{meta_charset}</head><body>')
	for group, transcript in sorted(by_group.items()):
		f.write(f'<h1>{group}</h1>')
		f.write('<table>')
		random.seed(1)
		random.shuffle(transcript)
		for t in transcript[:K]:
			try:
				audio_path = os.path.join(args.dataset_root, t['audio_path'])
			except:
				f.write('<tr><td>file not found: {audio_path}</td></tr>'.format(**t))
				continue
			f.write(
				'<tr><td>{audio_path}</td><td>{fmt_audio(audio_path)}</td><td>{ref}</td></tr>\n'.format(
					encoded = encoded, **t
				)
			)
		f.write('</table>')

	return iutput_path


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


def fmt_alignment(transcript, ref = None, hyp = None, flat = False, tag = '<pre>', prefix = True):
	span = lambda word, t = None: '<span style="{style}" title="{fmt_alignment_error_type}">{word}</span>'.format(word = word, style = ('background-color:' + dict(ok = 'green', missing = 'red', missing_ref = 'darkred', typo_easy = 'lightgreen', typo_hard = 'pink')[t]) if t is not None else '', fmt_alignment_error_type = t)

	error_tag = lambda w: w.get('type') or w.get('error_tag')
	if flat:
		ref_ = transcript.get('ref', '')
		hyp_ = transcript.get('hyp', '')
	else:
		ref_ = ' '.join(span(w.get('ref', ''), 'ok' if error_tag(w) == 'ok' else None) for w in transcript)
		hyp_ = ' '.join(span(w.get('hyp', ''), error_tag(w)) for w in transcript)

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
	cmd.set_defaults(func = lambda *args, **kwargs: print(transcript(*args, **kwargs)))

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
	cmd.set_defaults(func = lambda *args, **kwargs: print(errors(*args, **kwargs)))

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
	cmd.add_argument('--lang', default = 'ru')
	cmd.set_defaults(func = lambda *args, **kwargs: print(logits(*args, **kwargs)))

	cmd = subparsers.add_parser('audiosample')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o', required = True)
	cmd.add_argument('--dataset-root', default = '')
	cmd.add_argument('-K', type = int, default = 10)
	cmd.set_defaults(func = lambda *args, **kwargs: print(audiosample(*args, **kwargs)))

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
