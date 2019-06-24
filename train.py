import argparse
import torch
import torch.utils.data

import data
import data.dataset

parser = argparse.ArgumentParser()
parser.add_argument('--train-data-path', default = '../open_stt_splits/audiobooks_train.csv.gz')
parser.add_argument('--sample-rate', type = int, default = 16000)
parser.add_argument('--window-size', type = float, default = 0.2)
parser.add_argument('--window-stride', type = float, default = 0.1)
parser.add_argument('--window', default = 'hann', choices = ['hann', 'hamming'])
parser.add_argument('--num-workers', type = int, default = 10)
args = parser.parse_args()

audio_conf = dict(
	sample_rate = args.sample_rate,
	window_size = args.window_size,
	window_stride = args.window_stride,
	window = args.window
)

train_dataset = data.dataset.SpectrogramDataset(audio_conf = audio_conf, data_path = args.train_data_path)
train_loader = torch.utils.data.DataLoader(train_dataset, num_workers = args.num_workers, collate_fn = data.dataset.collate_fn, pin_memory = True, shuffle = True)

for i, (inputs, targets, filenames, input_percentages, target_sizes) in enumerate(train_loader):
    print(i, inputs.shape)
    break
