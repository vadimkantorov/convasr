import os
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
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint, map_location = 'cpu')
sample_rate, window_size, window_stride, window, num_input_features = list(map(checkpoint['args'].get, ['sample_rate', 'window_size', 'window_stride', 'window'])) + [checkpoint['num_input_features']]

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
	signal, sample_rate = dataset.read_wav(audio_path, sample_rate = sample_rate)
	features = models.logfbank(signal, sample_rate, window_size, window_stride, window, num_input_features)
	logits, output_lengths = model(features.unsqueeze(0).to(args.device), input_lengths_fraction = torch.IntTensor([1.0]))
	log_probs = F.log_softmax(logits, dim = 1)
	transcript = labels.idx2str(decoder.decode(F.log_softmax(logits, dim = 1), output_lengths.tolist()))[0]

	print(args.checkpoint)
	print(os.path.basename(audio_path))
	print(transcript)

	if args.output_path:
		torch.save(dict(filename = os.path.basename(audio_path), audio_path = audio_path, transcript = transcript, features = features.cpu(), log_probs = log_probs[0].cpu()), os.path.join(args.output_path, os.path.basename(audio_path) + '.pt'))

	reference_path = audio_path + '.txt' 
	if os.path.exists(reference_path):
		reference = labels.normalize_text(open(reference_path).read())
		cer = metrics.cer(transcript, reference)
		print(f'CER: {cer:.02%}')
	print()
