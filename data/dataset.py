import csv
import gzip
import numpy as np
import torch.utils.data
import scipy.io.wavfile
import librosa

import data.labels

class SpectrogramDataset(torch.utils.data.Dataset):
	def __init__(self, audio_conf, data_path, labels_path = None, max_duration = 20):
		self.window_stride = audio_conf['window_stride']
		self.window_size = audio_conf['window_size']
		self.sample_rate = audio_conf['sample_rate']
		self.window = getattr(scipy.signal, audio_conf['window'])
		self.labels = data.labels.Labels(labels_path)
		self.transforms = []
		# TODO: max_duration

		if data_path.endswith('.gz'):
			self.ids = [(row[-2], row[-1], float(row[4])) for row in csv.reader(gzip.open(data_path, 'rt')) if float(row[4]) < max_duration]
		else:
			self.ids = [('sample_ok/sample_ok/' + row[0], row[-1], 5) for row in csv.reader(data_path) if 'wav' in row[0]]

	def __getitem__(self, index):
		audio_path, transcript, duration = self.ids[index]
		
		signal, sample_rate = read_wav(audio_path); 
		if sample_rate != self.sample_rate:
			signal, sample_rate = librosa.resample(y, sample_rate, self.sample_rate), self.sample_rate
		# TODO: apply self.transforms 
		spect = spectrogram(signal, sample_rate, self.window_size, self.window_stride, self.window)
		transcript = self.labels.parse(transcript)
		return spect, transcript, audio_path

	def __len__(self):
		return len(self.ids)

def collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    filenames = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        filenames.append(sample[2])
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, filenames, input_percentages, target_sizes

def spectrogram(signal, sample_rate, window_size, window_stride, window):
	n_fft = int(sample_rate * (window_size + 1e-8))
	win_length = n_fft
	hop_length = int(sample_rate * (window_stride + 1e-8))
	D = librosa.stft(signal.numpy(), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
	spect = np.abs(D)

	if spect.shape[0] < 161:
		spect.resize((161, *spect.shape[1:]))
		spect[81:] = spect[80:0:-1]
	spect = spect[:161]
	return torch.from_numpy(spect)

def read_wav(path, channel=-1):
	sample_rate, signal = scipy.io.wavfile.read(path)
	signal = torch.from_numpy(signal).to(torch.float32)
	abs_max = signal.abs().max()
	if abs_max > 0:
		signal *= 1. / abs_max
	
	if len(signal.shape) > 1:
		if signal.shape[1] == 1:
			signal = signal.squeeze()
		elif channel == -1:
			signal = signal.mean(1)  # multiple channels, average
		else:
			signal = signal[:, channel]  # multiple channels, average
		assert len(signal.shape) == 1
	return signal, sample_rate
