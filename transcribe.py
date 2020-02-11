#TODO:

#batch by vad

# if given transcript, always align
# support stereo inputs with or without vad, with or without alignment
# fix up timestamps by segment begin point
# figure out stft shift
# disable repeat deduplication
# fix 1.0400128364562988 11.540141105651855 cut
# mix vad output for filtering full output
# filter vad output

import os
import io
import time
import json
import base64
import argparse
import importlib
import torch
import torch.nn.functional as F
import datasets
import models
import metrics
import decoders
import ctc
import vad
import segmentation
import audio

@torch.no_grad()
def main(args):
	if args.output_path:
		os.makedirs(args.output_path, exist_ok = True)
	
	checkpoint = torch.load(args.checkpoint, map_location = 'cpu')
	args.sample_rate, args.window_size, args.window_stride, args.window, args.num_input_features = map(checkpoint['args'].get, ['sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features'])
	frontend = models.LogFilterBankFrontend(args.num_input_features, args.sample_rate, args.window_size, args.window_stride, args.window)
	labels = datasets.Labels(importlib.import_module(checkpoint['args']['lang']), name = 'char')
	model = getattr(models, checkpoint['args']['model'])(args.num_input_features, [len(labels)], frontend = frontend, dict = lambda logits, log_probs, output_lengths, **kwargs: (logits[0], output_lengths[0]))
	model.load_state_dict(checkpoint['model_state_dict'], strict = False)
	model = model.to(args.device)
	model.eval()
	model.fuse_conv_bn_eval()
	#model = models.data_parallel(model)

	decoder = decoders.GreedyDecoder() if args.decoder == 'GreedyDecoder' else decoders.BeamSearchDecoder(labels, lm_path = args.lm, beam_width = args.beam_width, beam_alpha = args.beam_alpha, beam_beta = args.beam_beta, num_workers = args.num_workers, topk = args.decoder_topk)
	
	audio_paths = [p for f in args.data_path for p in ([os.path.join(f, g) for g in os.listdir(f)] if os.path.isdir(f) else [f]) if os.path.isfile(p) and any(map(p.endswith, args.ext))]

	val_dataset = datasets.AudioTextDataset(audio_paths, [labels], args.sample_rate, frontend = None, segmented = True, time_padding_multiple = args.batch_time_padding_multiple, vad_options = dict(window_size = args.window_size, aggressiveness = args.vad, window_size_dilate = args.window_size_dilate))
	val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size = None, collate_fn = val_dataset.collate_fn)

	for meta, x, xlen, y, ylen in val_data_loader:
		audio_path = meta[0]['audio_path']
		ref_path = audio_path + '.txt'
		ref = labels.normalize_text(open(ref_path).read()) if os.path.exists(ref_path) else ''
		
		tic = time.time()
		log_probs, output_lengths = model(x.to(args.device), xlen.to(args.device), y = y.to(args.device), ylen = ylen.to(args.device))

		speech = vad.detect_speech(x.squeeze(1), args.sample_rate, args.window_size, aggressiveness = args.vad, window_size_dilate = args.window_size_dilate)
		speech = vad.upsample(speech, log_probs)
		
		log_probs.masked_fill_(models.silence_space_mask(log_probs, speech, space_idx = labels.space_idx, blank_idx = labels.blank_idx), float('-inf'))
		decoded = decoder.decode(log_probs, output_lengths)
		
		print(os.path.basename(audio_path))
		print('Input time steps:', log_probs.shape[-1], '| target time steps:', y.shape[-1])
		print('Time: audio {audio:.02f} sec | processing {processing:.02f} sec'.format(audio = sum(t['duration'] for t in meta), processing = time.time() - tic))

		ts = torch.linspace(0, 1, steps = log_probs.shape[-1]) * (x.shape[-1] / args.sample_rate)
		segments = [[[], labels.decode(d, ts + t['begin'], channel = t['channel'], replace_blank = True, replace_repeat = True, replace_space = False)] for t, d in zip(meta, decoded)]
		hyp = ' '.join(t['word'] for r, h in segments for t in h)
		if args.output_path:
			open(os.path.join(args.output_path, os.path.basename(audio_path) + '.txt'), 'w').write(hyp)
		if args.verbose:
			print('HYP:', hyp)

		if ref and args.align:
			print('CER: {cer:.02%}'.format(cer = metrics.cer(hyp, ref)))
			tic = time.time()
			#TODO: collapse log_probs, y and ts
			alignment = ctc.ctc_loss_(log_probs.permute(2, 0, 1), y.long(), output_lengths, ylen, blank = labels.blank_idx, alignment = True)
			print('Alignment time: {:.02f} sec'.format(time.time() - tic))
			for i in range(len(y)):
				segments[i][-2] = labels.decode(y[i, :ylen[i]].tolist(), ts, alignment[i])
		
		segments = sum([list(segmentation.resegment(*s, ws = s[-2] or s[-1], max_segment_seconds = args.max_segment_seconds)) for s in segments], [])
		segments = segmentation.sort(segments)
		print(html_report(os.path.join(args.output_path, os.path.basename(audio_path) + '.html'), segments, audio_path, args.sample_rate))

def html_report(html_path, segments, audio_path, sample_rate):
	signal, sample_rate = audio.read_audio(audio_path, sample_rate = sample_rate, mono = False, dtype = torch.int16, normalize = False)
	with open(html_path, 'w') as html:
		fmt_link = lambda word, channel, begin, end, i = '', j = '': f'<a onclick="return play({channel},{begin},{end})" title="#{channel}: {begin:.04f} - {end:.04f} | {i} - {j}" href="#" target="_blank">' + (word if isinstance(word, str) else f'{begin:.02f}' if word == 0 else f'{end:.02f}' if word == 1 else f'{end - begin:.02f}') + '</a>'
		fmt_words = lambda rh: rh if isinstance(rh, str) else ' '.join(fmt_link(**w) for w in rh) if len(rh) > 0 and isinstance(rh[0], dict) else ' '.join(rh)
		fmt_begin_end = 'data-begin="{begin}" data-end="{end}"'.format

		html.write('<html><head><meta charset="UTF-8"><style>.m0{margin:0px} .top{vertical-align:top} .channel0{background-color:violet} .channel1{background-color:lightblue} .reference{opacity:0.4} .channel{margin:0px}</style></head><body>')
		html.write(f'<div style="overflow:auto"><h4 style="float:left">{os.path.basename(audio_path)}</h4><h5 style="float:right">0.000000</h5></div>')
		html.writelines(f'<figure class="m0"><figcaption>channel #{c}:</figcaption><audio ontimeupdate="ontimeupdate(event)" id="audio{c}" style="width:100%" controls src="data:audio/wav;base64,{base64.b64encode(wav).decode()}"></audio></figure>' for c, wav in enumerate(audio.write_audio(io.BytesIO(), signal[channel], sample_rate).getvalue() for channel in ([0, 1] if len(signal) == 2 else []) + [...]))
		html.write(f'''<pre class="channel"><h3 class="channel0 channel">hyp #0:<span></span></h3></pre><pre class="channel"><h3 class="channel0 reference channel">ref #0:<span></span></h3></pre><pre class="channel" style="margin-top: 10px"><h3 class="channel1 channel">hyp #1:<span></span></h3></pre><pre class="channel"><h3 class="channel1 reference channel">ref #1:<span></span></h3></pre><hr/>
		<table style="width:100%"><thead><th>begin</th><th>end</th><th>dur</th><th style="width:50%">hyp</th><th style="width:50%">ref</th><th>begin</th><th>end</th><th>dur</th></tr></thead><tbody>''')
		html.writelines(f'<tr class="channel{c}"><td class="top">{fmt_link(0, c, **segmentation.summary(h))}</td><td class="top">{fmt_link(1, c, **segmentation.summary(h))}</td><td class="top">{fmt_link(2, c, **segmentation.summary(h))}</td><td class="top hyp" data-channel="{c}" {fmt_begin_end(**segmentation.summary(h))}>{fmt_words(h)}</td>' + (f'<td class="top reference ref" data-channel="{c}" {fmt_begin_end(**segmentation.summary(r))}>{fmt_words(r)}</td><td class="top">{fmt_link(0,c, **segmentation.summary(r))}</td><td class="top">{fmt_link(1, c, **segmentation.summary(r))}</td><td class="top">{fmt_link(2, c, **segmentation.summary(r))}</td>' if r else '<td></td>' * 4) + f'</tr>' for r, h in segments for c in [(r + h)[0]['channel']])
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

			function ontimeupdate(evt)
			{
				const time = evt.target.currentTime, endtime = evt.target.dataset.endTime;
				if(time > endtime)
					return evt.target.pause();

				document.querySelector('h5').innerText = time.toString();
				const [spanhyp0, spanref0, spanhyp1, spanref1] = document.querySelectorAll('span');
				[spanhyp0.innerText, spanref0.innerText, spanhyp1.innerText, spanref1.innerText] = [subtitle(hyp_segments, time, 0), subtitle(ref_segments, time, 0), subtitle(hyp_segments, time, 1), subtitle(ref_segments, time, 1)];
			}

			const make_segment = td => [td.innerText, td.dataset.channel, td.dataset.begin, td.dataset.end];
			const hyp_segments = Array.from(document.querySelectorAll('.hyp')).map(make_segment), ref_segments = Array.from(document.querySelectorAll('.ref')).map(make_segment);
		</script></body></html>''')
	return html_path

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', required = True)
	parser.add_argument('--data-path', '-i', nargs = '+')
	parser.add_argument('--output-path', '-o')
	parser.add_argument('--device', default = 'cuda', choices = ['cpu', 'cuda'])
	parser.add_argument('--max-segment-seconds', type = float, default = 2)
	parser.add_argument('--num-workers', type = int, default = 32)
	parser.add_argument('--ext', default = ['wav', 'mp3'])
	parser.add_argument('--decoder', default = 'GreedyDecoder', choices = ['GreedyDecoder', 'BeamSearchDecoder'])
	parser.add_argument('--decoder-topk', type = int, default = 1)
	parser.add_argument('--beam-width', type = int, default = 5000)
	parser.add_argument('--beam-alpha', type = float, default = 0.3)
	parser.add_argument('--beam-beta', type = float, default = 1.0)
	parser.add_argument('--lm')
	parser.add_argument('--batch-time-padding-multiple', type = int, default = 128)
	parser.add_argument('--vad', type = int, choices = [0, 1, 2, 3], default = False, nargs = '?')
	parser.add_argument('--align', action = 'store_true')
	parser.add_argument('--verbose', action = 'store_true')
	parser.add_argument('--stereo', action = 'store_true')
	parser.add_argument('--window-size-dilate', type = float, default = 1.0)
	args = parser.parse_args()
	args.vad = args.vad if isinstance(args.vad, int) else 3
	
	main(args)
