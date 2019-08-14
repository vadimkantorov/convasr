import argparse
import importlib
import dataset
import decoders
import models

parser = argparse.ArgumentParser()
parser.add_argument('--num-input-features', type = int, default = 64)
parser.add_argument('--sample-rate', type = int, default = 16000)
parser.add_argument('--window-size', type = float, default = 0.02)
parser.add_argument('--window-stride', type = float, default = 0.01)
parser.add_argument('--window', default = 'hann', choices = ['hann', 'hamming'])
parser.add_argument('--model', default = 'Wav2LetterRu')
parser.add_argument('--checkpoint', required = True)
parser.add_argument('-i', '--audio-path', required = True)
parser.add_argument('--lang', default = 'ru')
parser.add_argument('--device', default = 'cuda', choices = ['cpu', 'cuda']) 
args = parser.parse_args()

labels = dataset.Labels(importlib.import_module(args.lang))
model = getattr(models, args.model)(num_classes = len(labels), num_input_features = args.num_input_features)
models.load_checkpoint(args.checkpoint, model)
model = model.to(args.device)
decoder = decoders.GreedyDecoder(labels.char_labels)
model.eval()
torch.set_grad_enabled(False)

signal, sample_rate = read_wav(args.audio_path, sample_rate = args.sample_rate)
features = logfbank(signal, args.sample_rate, args.window_size, args.window_stride, args.window, args.num_input_features)

inputs = features.unsqueeze(0)
input_lengths = torch.IntTensor([[spect.shape[-1]]])
logits = model(inputs.to(args.device), input_lengths)
output_lengths = models.compute_output_lengths(model, input_lengths)
idx = decoder.decode(F.softmax(logits, dim = 1).permute(0, 2, 1), output_lengths)

print(labels.idx2str(decoded_output))
