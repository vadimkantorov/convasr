import argparse
import importlib

import dataset
import decoder
import model
from model import load_checkpoint, save_checkpoint

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

lang = importlib.import_module(args.lang)
labels = dataset.Labels(lang.LABELS, preprocess_text = lang.preprocess_text, preprocess_word = lang.preprocess_word)
model = model.Speech2TextModel(getattr(model, args.model)(num_classes = len(labels.char_labels), num_input_features = args.num_input_features))
load_checkpoint(model, args.checkpoint)
model = model.to(args.device)
decoder = decoder.GreedyDecoder(labels.char_labels)
model.eval()

torch.set_grad_enabled(False)

spect, transcript_dummy, audio_path_dummy = dataset.load_example(args.audio_path, transcript = '', sample_rate = args.sample_rate, window_size = args.window_size, window_stride = args.window_stride, num_input_features = args.num_input_features, window = args.window)
inputs = spect.unsqueeze(0)
input_sizes = torch.IntTensor([[spect.shape[-1]]])

logits, output_sizes = model(inputs.to(args.device), input_sizes)
decoded_output, decoded_offsets = decoder.decode(F.softmax(logits, dim = 1).permute(0, 2, 1), output_sizes)

print('HYP:', decoded_output)
