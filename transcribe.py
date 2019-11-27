import os
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

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required = True)
parser.add_argument('-i', '--data-path', required = True)
parser.add_argument('-o', '--output-path')
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
parser.add_argument('--vad', type = int, choices = [0, 1, 2, 3], default = False, nargs = '?')
args = parser.parse_args()

args.output_path = args.output_path or args.data_path

vad = webrtcvad.Vad()
vad.set_mode(args.vad if isinstance(args.vad, int) else 3)

checkpoint = torch.load(args.checkpoint, map_location = 'cpu')
sample_rate, window_size, window_stride, window, num_input_features = map(checkpoint['args'].get, ['sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features'])

labels = dataset.Labels(importlib.import_module(checkpoint['lang']), name = 'char')
model = getattr(models, checkpoint['model'])(num_input_features = num_input_features, num_classes = [len(labels)], dict = lambda logits, output_lengths: (logits, output_lengths))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(args.device)
model.eval()
model.fuse_conv_bn_eval()
#model = torch.nn.DataParallel(model)

decoder = decoders.GreedyDecoder() if args.decoder == 'GreedyDecoder' else decoders.BeamSearchDecoder(labels, lm_path = args.lm, beam_width = args.beam_width, beam_alpha = args.beam_alpha, beam_beta = args.beam_beta, num_workers = args.num_workers, topk = args.decoder_topk)
torch.set_grad_enabled(False)

os.makedirs(args.output_path, exist_ok = True)

audio_paths = [args.data_path] if os.path.isfile(args.data_path) else [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if any(map(f.endswith, args.ext))]
for audio_path in audio_paths:
	reference_path = audio_path + '.json'
	reference = json.load(open(reference_path)) if os.path.exists(reference_path) else []
	batch, cutpoints = [], []
	if reference:
		signal, sample_rate = dataset.read_wav(audio_path, sample_rate = sample_rate, stereo = True)
		for b, e, channel, ref in [map(r.get, ['begin', 'end', 'channel', 'ref']) for r in reference]:
			cutpoints.append((b, e, channel))
			batch.append((os.path.basename(os.path.dirname(audio_path)), audio_path, ref, signal[None, int(b * sample_rate):int(e * sample_rate), channel], torch.IntTensor()))
	else:
		signal, sample_rate = dataset.read_wav(audio_path, sample_rate = sample_rate, stereo = True, normalize = False, dtype = torch.int16)
		for channel, signal_ in enumerate(signal.t()):
			chunks = dataset.remove_silence(vad, signal_, sample_rate, window_size) if args.vad is not False else [(0, len(signal))]
			signal_ = models.normalize_signal(signal_)
			cutpoints.extend((b / sample_rate, e / sample_rate, channel) for b, e in chunks)
			batch.extend((os.path.basename(os.path.dirname(audio_path)), audio_path, '', signal_[None, b:e], torch.IntTensor()) for b, e in chunks)

	dataset_name_, audio_path_, reference_, input_, input_lengths_fraction_, targets_, target_length_ = dataset.collate_fn(batch)
	features = models.logfbank(input_.squeeze(1), sample_rate, window_size, window_stride, window, num_input_features)
	log_probs, output_lengths = model(features.to(args.device, non_blocking = True), input_lengths_fraction_.to(args.device, non_blocking = True)
	
	decoded = decoder.decode(log_probs, output_lengths)

	transcript = labels.postprocess_transcript(' '.join(map(labels.decode, decoded)))
	print(args.checkpoint, os.path.basename(audio_path))
	print('HYP:', transcript, '\n')
	open(os.path.join(args.output_path, os.path.basename(audio_path) + '.txt'), 'w').write(transcript)
	
	def segment_transcript(b, e, idx, labels):
		sec = lambda k: k / len(idx) * (e - b)
		i = 0
		for j in range(1, 1 + len(idx)):
			if j == len(idx) or (idx[j] == labels.space_idx and sec(j - 1) - sec(i) > args.max_segment_seconds):
				yield (b + sec(i), b + sec(j - 1), labels.postprocess_transcript(labels.decode(idx[i:j])))
				i = j + 1

	segments = [[b_, e_, t_, r, c] for (b, e, c), d, r in zip(cutpoints, decoded, reference_) for b_, e_, t_ in segment_transcript(b, e, d, labels)] if args.vad is not False else [[b, e, labels.postprocess_transcript(labels.decode(d)), r, c] for (b, e, c), d, r in zip(cutpoints, decoded, reference_)]
	
	reference_path = audio_path + '.txt' 
	if os.path.exists(reference_path):
		reference = labels.postprocess_transcript(labels.normalize_text(open(reference_path).read()))
		cer = metrics.cer(transcript, reference)
		print(f'CER: {cer:.02%}')

	html = open(os.path.join(args.output_path, os.path.basename(audio_path) + '.html'), 'w')
	html.write('<html><head><meta charset="UTF-8"><style>.channel0{background-color:violet} .channel1{background-color:lightblue} .reference{opacity:0.4} .channel{margin:0px}</style></head><body>')
	html.write(f'<h4>{os.path.basename(audio_path)}</h4>')
	html.write('<audio style="width:100%" controls src="data:audio/wav;base64,{encoded}"></audio>'.format(encoded = base64.b64encode(open(audio_path, 'rb').read()).decode()))
	html.write(f'<h3 class="channel0 channel">hyp #0:<span></span></h3><h3 class="channel0 reference channel">ref #0:<span></span></h3><h3 class="channel1 channel">hyp #1:<span></span></h3><h3 class="channel1 reference channel">ref #1:<span></span></h3><hr/>')
	html.write('<table style="width:100%"><thead><th>begin</th><th>end</th><th>transcript</th></tr></thead><tbody>')
	html.write(''.join(f'<tr class="channel{c}"><td>{b:.02f}</td><td>{e:.02f}</td><td><a onclick="play({b:.02f}, {e:.02f}); return false;" href="#" target="_blank">{t}</a></td>' + (f'<td>{r}</td></tr><tr class="channel{c} reference"><td></td><td></td><td>{r}</td></tr>' if r else '') for b, e, t, r, c in sorted(segments)))
	html.write('</tbody></table>')
	html.write('''<script>
		const segments = SEGMENTS;

		function play(begin, end)
		{
			const audio = document.querySelector('audio');
			audio.currentTime = begin;
			audio.dataset.endTime = end;
			audio.play();
		};

		document.querySelector('audio').ontimeupdate = (evt) =>
		{
			const time = evt.target.currentTime;
			const endtime = evt.target.dataset.endTime;

			const [spanhyp0, spanref0, spanhyp1, spanref1] = document.querySelectorAll('span');
			const [begin0, end0, transcript0, reference0, channel0] = segments.find(([begin, end, transcript, reference, channel]) => channel == 0 && begin <= time && time <= end) || [null, null, '', '', 0];
			const [begin1, end1, transcript1, reference1, channel1] = segments.find(([begin, end, transcript, reference, channel]) => channel == 1 && begin <= time && time <= end) || [null, null, '', '', 1];
			[spanhyp0.innerText, spanhyp1.innerText] = [transcript0, transcript1];
			[spanref0.innerText, spanref1.innerText] = [reference0, reference1];
		};
	</script>'''.replace('SEGMENTS', json.dumps(segments)))
	html.write('</body></html>')
