import os
import time
import json
import base64
import argparse
import webrtcvad
import importlib
import torch
import torch.nn.functional as F
import dataset
import models
import metrics
import decoders
import ctc

def segment_transcript(labels, idx, b, e, max_segment_seconds):
	sec = lambda k: k / len(idx) * (e - b)
	i = 0
	for j in range(1, 1 + len(idx)):
		if j == len(idx) or (idx[j] == labels.space_idx and sec(j - 1) - sec(i) > max_segment_seconds):
			yield (b + sec(i), b + sec(j - 1), labels.postprocess_transcript(labels.decode(idx[i:j])[0]))
			i = j + 1

def resegment(c, b, e, r, h, max_segment_seconds):
	rh_ = lambda rh, i, w, first, last: [(k, u) for k, u in enumerate(rh) if (first or i is None or u['begin'] >= rh[i]['end']) and (last or u['end'] <= w['end'])]
	rh, rhk = r, 0
	i = [None, None]
	for j, w in enumerate(rh):
		if j == len(rh) - 1 or w['end'] - rh[i[rhk] or 0]['end'] > max_segment_seconds:
			first_last = dict(first = i[0] is None, last = j == len(rh) - 1)
			rk, r_ = zip(*rh_(r, i[0], rh[j], **first_last))
			hk, h_ = zip(*rh_(h, i[1], rh[j], **first_last))
			i = (rk[-1], hk[-1])
			yield [c, min(w['begin'] for w_ in [r_, h_] for w in w_), max(w['end'] for w_ in [r_, h_] for w in w_), r_, h_]

def main(args):
	os.makedirs(args.output_path, exist_ok = True)
	
	vad = webrtcvad.Vad()
	vad.set_mode(args.vad if isinstance(args.vad, int) else 3)

	torch.set_grad_enabled(False)

	checkpoint = torch.load(args.checkpoint, map_location = 'cpu')
	args.sample_rate, args.window_size, args.window_stride, args.window, args.num_input_features = map(checkpoint['args'].get, ['sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features'])
	batch_collater = dataset.BatchCollater(args.batch_time_padding_multiple)
	frontend = models.LogFilterBankFrontend(args.num_input_features, args.sample_rate, args.window_size, args.window_stride, args.window)
	labels = dataset.Labels(importlib.import_module(checkpoint['args']['lang']), name = 'char')
	model = getattr(models, checkpoint['args']['model'])(args.num_input_features, [len(labels)], frontend = frontend, dict = lambda logits, log_probs, output_lengths, **kwargs: (logits[0], output_lengths[0]))
	model.load_state_dict(checkpoint['model_state_dict'], strict = False) #TODO: figure out problems with frontend
	model = model.to(args.device)
	model.eval()
	model.fuse_conv_bn_eval()
	#model = models.data_parallel(model)

	decoder = decoders.GreedyDecoder() if args.decoder == 'GreedyDecoder' else decoders.BeamSearchDecoder(labels, lm_path = args.lm, beam_width = args.beam_width, beam_alpha = args.beam_alpha, beam_beta = args.beam_beta, num_workers = args.num_workers, topk = args.decoder_topk)
	
	example = lambda audio_path, signal, b, e, sample_rate, channel, ref_normalized, targets: (os.path.basename(os.path.dirname(audio_path)), audio_path, ref_normalized, signal[None, int(b * sample_rate):int(e * sample_rate), channel], targets)	

	audio_paths = [args.data_path] if os.path.isfile(args.data_path) else [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if any(map(f.endswith, args.ext))]
	for audio_path in audio_paths:
		batch, cutpoints = [], []
		ref_path, transcript_path = audio_path + '.txt', audio_path + '.json'
		if os.path.exists(transcript_path):
			signal_normalized, sample_rate = dataset.read_audio(audio_path, sample_rate = args.sample_rate, stereo = True, normalize = True, dtype = torch.float32)
			transcript = json.load(open(transcript_path))
			for b, e, channel, ref in [map(r.get, ['begin', 'end', 'channel', 'ref']) for r in transcript]:
				cutpoints.append((b, e, channel))
				batch.append(example(audio_path, signal_normalized, b, e, sample_rate, channel, *labels.encode(ref)))
		else:
			signal, sample_rate = dataset.read_audio(audio_path, sample_rate = args.sample_rate, stereo = True, normalize = False, dtype = torch.int16)
			signal = signal[:len(signal) // 10]
			signal_normalized = models.normalize_signal(signal, dim = 0)
			ref = labels.postprocess_transcript(labels.normalize_text(open(ref_path).read())) if os.path.exists(ref_path) else ''
			ref = ref[:len(ref) // 10]
			for channel, signal_ in enumerate(signal.t()):
				chunks = dataset.remove_silence(vad, signal_, sample_rate, window_size) if args.vad is not False else [(0, len(signal) / sample_rate)]
				cutpoints.extend((b, e, channel) for b, e in chunks)
				batch.extend(example(audio_path, signal_normalized, b, e, sample_rate, channel, *labels.encode(ref)) for b, e in chunks)

		tic = time.time()
		_, _, ref, x, xlen, y, ylen = batch_collater(batch)
		x, xlen, y, ylen = [t.to(args.device) for t in [x, xlen, y, ylen]]
		x, y, ylen = x.squeeze(1), y.squeeze(1), ylen.squeeze(1)
		log_probs, output_lengths = model(x, xlen)
		decoded = decoder.decode(log_probs, output_lengths)
		
		print(args.checkpoint, os.path.basename(audio_path))
		print('Time: audio {audio:.02f} sec | voice {voice:.02f} sec | processing {processing:.02f} sec'.format(audio = signal.numel() / sample_rate, voice = sum(e - b for b, e, c in cutpoints), processing = time.time() - tic))

		#segments = [[c, b_, e_, r, t_] for (b, e, c), d, r in zip(cutpoints, decoded, ref) for b_, e_, t_ in segment_transcript(labels, d, b, e, args.max_segment_seconds)] if args.vad is not False else [[c, b, e, r, labels.postprocess_transcript(labels.decode(d)[0])] for (b, e, c), d, r in zip(cutpoints, decoded, ref)]
		
		ts = torch.linspace(0, 1, steps = log_probs.shape[-1]) * (x.shape[-1] / sample_rate)
		segments = [[c, b, e, r, labels.decode(d, ts, replace_blank = True, replace_repeat = True)] for (b, e, c), d, r in zip(cutpoints, decoded, ref)]

		ref = ' '.join(ref)
		hyp = ' '.join(w['word'] for c, b, e, r, h in segments for w in h)
		open(os.path.join(args.output_path, os.path.basename(audio_path) + '.txt'), 'w').write(hyp)
		if args.verbose:
			print('HYP:', hyp)
		
		if ref:
			cer = metrics.cer(hyp, ref)#, edit_distance = metrics.levenshtein)
			if args.align:
				print('Input time steps:', log_probs.shape[-1], '| Target time steps:', y.shape[-1])
				tic = time.time()
				alignment = ctc.ctc_loss(log_probs.permute(2, 0, 1).half(), y.long(), output_lengths, ylen, blank = labels.blank_idx, alignment = True).argmax(dim = 0)
				print('Alignment time: {:.02f} sec'.format(time.time() - tic))
				for i in range(len(y)):
					segments[i][-2] = labels.decode(y[i].tolist(), ts[alignment[i]])
			print(f'CER: {cer:.02%}')

		segments = sum([list(resegment(*s, max_segment_seconds = args.max_segment_seconds)) for s in segments], [])

		html_path = os.path.join(args.output_path, os.path.basename(audio_path) + '.html')
		with open(html_path, 'w') as html:
			fmt_link = lambda begin, end, word = None: f'<a onclick="return play({begin},{end})" href="#" target="_blank">' + (word if isinstance(word, str) else f'{word:.02f}') + '</a>'
			fmt_words = lambda h: h if isinstance(h, str) else ' '.join(fmt_link(**w) for w in h) if len(h) > 0 and isinstance(h[0], dict) else ' '.join(h)
			
			html.write('<html><head><meta charset="UTF-8"><style>.top{vertical-align:top} .channel0{background-color:violet} .channel1{background-color:lightblue} .reference{opacity:0.4} .channel{margin:0px}</style></head><body>')
			html.write(f'<h4>{os.path.basename(audio_path)}</h4>')
			html.write('<audio style="width:100%" controls src="data:audio/wav;base64,{encoded}"></audio>'.format(encoded = base64.b64encode(open(audio_path, 'rb').read()).decode()))
			html.write(f'<h3 class="channel0 channel">hyp #0:<span></span></h3><h3 class="channel0 reference channel">ref #0:<span></span></h3><h3 class="channel1 channel">hyp #1:<span></span></h3><h3 class="channel1 reference channel">ref #1:<span></span></h3><hr/>')
			html.write('<table style="width:100%"><thead><th>begin</th><th>end</th><th style="width:50%">hyp</th><th style="width:50%">ref</th></tr></thead><tbody>')
			html.write(''.join(f'<tr class="channel{c}"><td class="top">{fmt_link(b, e, b)}</td><td class="top">{fmt_link(b, e, e)}</td><td class="top">{fmt_words(h)}</td>' + (f'<td class="top reference">{fmt_words(r)}</td>' if r else '') + '</tr>' for c, b, e, r, h in sorted(segments, key = lambda s: s[:3])))
			html.write('</tbody></table>')
			html.write('''<script>
				const segments = SEGMENTS;

				function play(begin, end)
				{
					const audio = document.querySelector('audio');
					audio.currentTime = begin;
					audio.dataset.endTime = end;
					audio.play();
					return false;
				};

				document.querySelector('audio').ontimeupdate = evt =>
				{
					const time = evt.target.currentTime;
					const endtime = evt.target.dataset.endTime;

					const [spanhyp0, spanref0, spanhyp1, spanref1] = document.querySelectorAll('span');
					const [channel0, begin0, end0, ref0, hyp0] = segments.find(([channel, begin, end, ref, hyp]) => channel == 0 && begin <= time && time <= end) || [0, null, null, [], []];
					const [channel1, begin1, end1, ref1, hyp1] = segments.find(([channel, begin, end, ref, hyp]) => channel == 1 && begin <= time && time <= end) || [1, null, null, [], []];
					const fmt_words = words => words.map(w => w['word']).join(' ');
					//[spanhyp0.innerText, spanhyp1.innerText, spanref0.innerText, spanref1.innerText] = [fmt_words(hyp0), fmt_words(hyp1), fmt_words(ref0), fmt_words(ref1)];

					if(time > endtime)
						evt.target.pause();
				};
			</script>'''.replace('SEGMENTS', json.dumps(segments)))
			html.write('</body></html>')
		print(html_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', required = True)
	parser.add_argument('-i', '--data-path', required = True)
	parser.add_argument('-o', '--output-path', required = True)
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
	args = parser.parse_args()
	main(args)
