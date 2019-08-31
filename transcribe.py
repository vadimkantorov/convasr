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
parser.add_argument('--num-input-features', type = int, default = 64)
parser.add_argument('--sample-rate', type = int, default = 8000)
parser.add_argument('--model', default = 'Wav2LetterRu')
parser.add_argument('--checkpoint', required = True)
parser.add_argument('-i', '--data-path', required = True)
parser.add_argument('--device', default = 'cuda', choices = ['cpu', 'cuda']) 
parser.add_argument('--verbose', action = 'store_true')
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint, map_location = 'cpu')
sample_rate, window_size, window_stride, window, num_input_features = list(map(checkpoint['args'].get, ['sample_rate', 'window_size', 'window_stride', 'window'])) + [checkpoint['num_input_features']]

labels = dataset.Labels(importlib.import_module(checkpoint['lang']))
model = getattr(models, checkpoint['model'])(num_classes = len(labels), num_input_features = num_input_features)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(args.device)
decoder = decoders.GreedyDecoder(labels)
model.eval()
torch.set_grad_enabled(False)

audio_paths = [args.data_path] if os.path.isfile(args.data_path) else [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.endswith('.wav')]

for audio_path in audio_paths:
	reference_path, transcript_path = audio_path + '.txt', audio_path + '.transcript.txt'

	signal, sample_rate = dataset.read_wav(audio_path, sample_rate = sample_rate)
	features = models.logfbank(signal, sample_rate, window_size, window_stride, window, num_input_features)
	inputs = features.unsqueeze(0)
	input_lengths = torch.IntTensor([[features.shape[-1]]])
	logits = model(inputs.to(args.device), input_lengths)
	log_probs = F.log_softmax(logits, dim = 1)
	output_lengths = models.compute_output_lengths(model, input_lengths)
	entropy = float(models.entropy(log_probs, output_lengths, dim = 1).mean())
	transcript = labels.idx2str(decoder.decode(F.log_softmax(logits, dim = 1), output_lengths.tolist()))[0]
	open(transcript_path, 'w').write(transcript)

	print(os.path.basename(audio_path))
	if args.verbose:
		print(transcript)
	print(f'Entropy: {entropy:.2f}')
	if os.path.exists(reference_path):
		reference = labels.normalize_text(open(reference_path).read())
		cer = metrics.cer(transcript, reference)
		print(f'CER: {cer:.02%}')
	print()
