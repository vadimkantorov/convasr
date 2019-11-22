import os
import json
import base64
import argparse
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
args = parser.parse_args()

args.output_path = args.output_path or args.data_path

checkpoint = torch.load(args.checkpoint, map_location = 'cpu')
sample_rate, window_size, window_stride, window, num_input_features = map(checkpoint['args'].get, ['sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features'])

labels = dataset.Labels(importlib.import_module(checkpoint['lang']))
model = getattr(models, checkpoint['model'])(num_classes = len(labels), num_input_features = num_input_features)
MODEL_STRIDE = 2
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(args.device)
model.eval()
decoder = decoders.GreedyDecoder(labels) if args.decoder == 'GreedyDecoder' else decoders.BeamSearchDecoder(labels, lm_path = args.lm, beam_width = args.beam_width, beam_alpha = args.beam_alpha, beam_beta = args.beam_beta, num_workers = args.num_workers, topk = args.decoder_topk)
torch.set_grad_enabled(False)

os.makedirs(args.output_path, exist_ok = True)

audio_paths = [args.data_path] if os.path.isfile(args.data_path) else [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if any(map(f.endswith, args.ext))]
for audio_path in audio_paths:
	reference_path = audio_path + '.json'
	signal_, sample_rate = dataset.read_wav(audio_path, sample_rate = sample_rate, stereo = True)
	reference = json.load(open(reference_path)) if os.path.exists(reference_path) else None
	log_probs_ = []

	for channel, signal in enumerate(signal_.t()):
		features = models.logfbank(signal, sample_rate, window_size, window_stride, window, num_input_features)
		logits, output_lengths = model(features.unsqueeze(0).to(args.device), torch.tensor([1.0]).to(args.device))
		log_probs = F.log_softmax(logits, dim = 1)
		transcript = labels.decode(decoder.decode(log_probs, output_lengths)[0])
		transcript = labels.postprocess_transcript(transcript)
		log_probs_.extend(log_probs.cpu())

		print(args.checkpoint)
		print(os.path.basename(audio_path), 'channel#', channel)
		print(transcript)
		print()

		open(os.path.join(args.output_path, os.path.basename(audio_path) + f'.{channel}.txt'), 'w').write(transcript)
		
		#reference_path = audio_path + '.txt' 
		#if os.path.exists(reference_path):
		#	reference = labels.normalize_text(open(reference_path).read())
		#	cer = metrics.cer(transcript, reference)
		#	print(f'CER: {cer:.02%}')
		#print()

	replaceblankrepeat = lambda r: r.replace(labels.blank, '').replace('_', '')
	segments = []
	for channel, log_probs in enumerate(log_probs_):
		transcript = labels.idx2str(log_probs.argmax(dim = 0))
		if reference is None:
			begin, end = zip(*[(i*window_stride * MODEL_STRIDE, i*window_stride * MODEL_STRIDE + window_size) for i in range(log_probs.shape[-1])])
			k = 0
			for i in range(len(transcript) + 1):
				if(i > 0 and (i == len(transcript) or transcript[i] == ' ') and (i == len(transcript) or end[i - 1] - begin[k] > args.max_segment_seconds)):
					segments.append([begin[min(k, len(begin) - 1)], end[i - 1], replaceblankrepeat(transcript[min(k, len(transcript) - 1):i]), '', channel])
					k = i + 1
		else:
			for r in reference[channel]:
				begin, end = r['begin'], r['end']
				b, e = [int(t // window_stride // MODEL_STRIDE) for t in [begin, end]]
				segments.append([begin, end, replaceblankrepeat(transcript[b:e]), r['reference'], channel])

	html = open(os.path.join(args.output_path, os.path.basename(audio_path) + '.html'), 'w')
	html.write('<html><head><meta charset="UTF-8"><style>.channel0{background-color:violet} .channel1{background-color:lightblue} .reference{opacity:0.4} .channel{margin:0px}</style></head><body>')
	html.write(f'<h4>{os.path.basename(audio_path)}</h4>')
	encoded = base64.b64encode(open(audio_path, 'rb').read()).decode('utf-8').replace('\n', '')
	html.write(f'<audio style="width:100%" controls src="data:audio/wav;base64,{encoded}"></audio>')
	html.write(f'<h3 class="channel0 channel">transcript #0:<span></span></h3><h3 class="channel1 channel">transcript #1:<span></span></h3><h3 class="channel0 reference channel">reference #0:<span></span></h3><h3 class="channel1 reference channel">reference #1:<span></span></h3> <hr/>')
	html.write('<table><thead><th>begin</th><th>end</th><th>transcript</th></tr></thead><tbody>')
	html.write(''.join(f'<tr class="channel{c}"><td>{b:.02f}</td><td>{e:.02f}</td><td><a onclick="play({b:.02f}); return false;" href="#" target="_blank">{t}</a></td>' + (f'<td>{r}</td></tr><tr class="channel{c} reference"><td></td><td></td><td>{r}</td></tr>' if r else '') for b, e, t, r, c in sorted(segments)))
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
			const [spanhyp0, spanhyp1, spanref0, spanref1] = document.querySelectorAll('span');
			const time = evt.target.currentTime;
			const [begin0, end0, transcript0, reference0, channel0] = segments.find(([begin, end, transcript, reference, channel]) => channel == 0 && begin <= time && time <= end) || [null, null, '', '', 0];
			const [begin1, end1, transcript1, reference1, channel1] = segments.find(([begin, end, transcript, reference, channel]) => channel == 1 && begin <= time && time <= end) || [null, null, '', '', 1];

			spanhyp0.innerText = transcript0;
			spanhyp1.innerText = transcript1;
			spanref0.innerText = reference0;
			spanref1.innerText = reference1;
		};
	</script>'''.replace('SEGMENTS', repr(segments)))
	html.write('</body></html>')
