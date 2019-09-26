import os
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
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint, map_location = 'cpu')
sample_rate, window_size, window_stride, window, num_input_features = map(checkpoint['args'].get, ['sample_rate', 'window_size', 'window_stride', 'window', 'num_input_features'])

labels = dataset.Labels(importlib.import_module(checkpoint['lang']))
model = getattr(models, checkpoint['model'])(num_classes = len(labels), num_input_features = num_input_features)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(args.device)
model.eval()
decoder = decoders.GreedyDecoder(labels)
torch.set_grad_enabled(False)

if args.output_path:
	os.makedirs(args.output_path, exist_ok = True)

audio_paths = [args.data_path] if os.path.isfile(args.data_path) else [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.endswith('.wav')]
for audio_path in audio_paths:
	signal_, sample_rate = dataset.read_wav(audio_path, sample_rate = sample_rate, stereo = True)
	log_probs_ = []

	for channel, signal in enumerate(signal_.t()):
		features = models.logfbank(signal, sample_rate, window_size, window_stride, window, num_input_features)
		logits, output_lengths = model(features.unsqueeze(0).to(args.device), input_lengths_fraction = torch.IntTensor([1.0]))
		log_probs = F.log_softmax(logits, dim = 1)
		transcript = labels.idx2str(decoder.decode(F.log_softmax(logits, dim = 1), output_lengths.tolist())[0])
		log_probs_.extend(log_probs)

		print(args.checkpoint)
		print(os.path.basename(audio_path), 'channel#', channel)
		print(transcript)
		print()

	#reference_path = audio_path + '.txt' 
	#if os.path.exists(reference_path):
	#	reference = labels.normalize_text(open(reference_path).read())
	#	cer = metrics.cer(transcript, reference)
	#	print(f'CER: {cer:.02%}')
	#print()

	if args.output_path:
		segments = []
		for channel, log_probs in enumerate(log_probs_):
			MODEL_STRIDE = 2
			begin, end = map(list, zip(*[(i*window_stride * MODEL_STRIDE, i*window_stride * MODEL_STRIDE + window_size) for i in range(log_probs.shape[-1])]))
			transcript = labels.idx2str(log_probs.argmax(dim = 0), blank = labels.blank, repeat = '_')
			k = 0
			for i in range(len(transcript) + 1):
				if(i > 0 and (i == len(transcript) or transcript[i] == ' ') and (i == len(transcript) or end[i - 1] - begin[k] > args.max_segment_seconds)):
					segments.append([channel, begin[k], end[i - 1], transcript[k:i].replace(labels.blank, '').replace('_', '') ])
					k = i + 1
	
		html = open(os.path.join(args.output_path, os.path.basename(audio_path) + '.html'), 'w')
		html.write('<html><head><meta charset="UTF-8"><style>.channel0{background-color:violet} .channel1{background-color:lightblue}</style></head><body>')
		html.write(f'<h4>{os.path.basename(audio_path)}</h4>')
		encoded = base64.b64encode(open(audio_path, 'rb').read()).decode('utf-8').replace('\n', '')
		html.write(f'<audio style="width:100%" controls src="data:audio/wav;base64,{encoded}"></audio>')
		html.write(f'<h3>transcript</h3><hr />')
		html.write('<table><thead><tr><th>#</th><th>from</th><th>to</th><th>transcript</th></tr></thead><tbody>')
		html.write(''.join(f'<tr class="channel{c}"><td><strong>{c}</strong></td><td>{b:.02}</td><td>{e:.02}</td><td><a onclick="play({b:.02}); return false;" href="#" target="_blank">{t}</a></td></tr>' for c, b, e, t in segments))
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
				const h3 = document.querySelector('h3');
				const time = evt.target.currentTime;
				const [channel, begin, end, transcript] = segments.find(segment => segment[1] <= time && time <= segment[2]);
				h3.className = `channel${channel}`;
				h3.innerText = transcript;
			};
		</script>'''.replace('SEGMENTS', repr(segments)))
		html.write('</body></html>')
