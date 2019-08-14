import os
import re
import csv
import gzip
import math
import random
import numpy as np
import torch.utils.data
import scipy.io.wavfile
import scipy.signal
import librosa

class SpectrogramDataset(torch.utils.data.Dataset):
	def __init__(self, data_or_path, sample_rate, window_size, window_stride, window, num_input_features, labels, waveform_transform = None, feature_transform = None, max_duration = 20, normalize_features = True):
		self.window_stride = window_stride
		self.window_size = window_size
		self.sample_rate = sample_rate
		self.window = window
		self.num_input_features = num_input_features
		self.labels = labels
		self.waveform_transform = waveform_transform
		self.feature_transform = feature_transform
		self.normalize_features = normalize_features
		self.ids = [(row[0], row[1], float(row[2]) if len(row) > 2 else -1) for row in csv.reader(gzip.open(data_or_path, 'rt') if data_or_path.endswith('.gz') else open(data_or_path)) if len(row) <= 2 or float(row[2]) < max_duration] if isinstance(data_or_path, str) else [d for d in data_or_path if d[-1] == -1 or d[-1] < max_duration]

	def __getitem__(self, index):
		audio_path, transcript, duration = self.ids[index]
		features, transcript, audio_path = load_example(audio_path, transcript, self.sample_rate, self.window_size, self.window_stride, self.window, self.num_input_features, self.labels.parse, waveform_transform = self.waveform_transform, feature_transform = self.feature_transform, normalize_features = self.normalize_features)
		return features, transcript, audio_path

	def __len__(self):
		return len(self.ids)

class BucketingSampler(torch.utils.data.Sampler):
	def __init__(self, data_source, batch_size=1):
		super(BucketingSampler, self).__init__(data_source)
		self.data_source = data_source
		ids = list(range(0, len(data_source)))
		self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
		self.batch_idx = 0

	def __iter__(self):
		for ids in self.bins[self.batch_idx:]:
			np.random.shuffle(ids)
			yield ids
		self.batch_idx = 0

	def __len__(self):
		return len(self.bins)

	def shuffle(self, epoch):
		np.random.shuffle(self.bins)

	def state_dict(self, batch_idx):
		return dict(bins = self.bins, batch_idx = batch_idx)

	def load_state_dict(self, state_dict):
		self.bins = state_dict['bins']
		self.batch_idx = state_dict['batch_idx']

class Labels(object):
	blank = '|'
	space = ' '

	def __init__(self, lang):
		self.idx2chr_ = lang.LABELS
		self.preprocess_text = lang.preprocess_text
		self.preprocess_word = lang.preprocess_word
		self.chr2idx_ = {l: i for i, l in enumerate(self.idx2chr_)}
		self.blank_idx = self.idx2chr_.find(self.blank)
		self.space_idx = self.idx2chr_.find(self.space)

	def find_words(self, text):
		text = re.sub(r'([^\W\d]+)2', r'\1', text)
		text = self.preprocess_text(text)
		words = re.findall(r'-?\d+|-?\d+-\w+|\w+', text)
		return list(filter(bool, (''.join([c for c in self.preprocess_word(w) if c.upper() in self.chr2idx_]).strip() for w in words)))

	def parse(self, text):
		if text.startswith('!clean:'):
			return ''.join(map(self.chr2idx, text.replace('!clean:', '', 1).strip()))
		chars = ' '.join(self.find_words(text)).upper().strip() or '*'
		return [self.chr2idx(c) if i == 0 or c != chars[i - 1] else self.chr2idx('2') for i, c in enumerate(chars)]

	def idx2str(self, idx):
		i2s = lambda i: ''.join(map(self.idx2chr, i))
		return list(map(i2s, idx)) if isinstance(idx[0], list) else i2s(idx)

	def chr2idx(self, chr):
		return self.chr2idx_[chr]

	def idx2chr(self, idx):
		return self.idx2chr_[idx]

	def __len__(self):
		return len(self.idx2chr_)

	def __str__(self):
		return self.idx2chr_

def unpack_targets(targets, target_sizes):
	unpacked = []
	offset = 0
	for size in target_sizes:
		unpacked.append(targets[offset:offset + size])
		offset += size
	return unpacked

def collate_fn(batch):
	duration_in_frames = lambda example: example[0].shape[-1]
	batch = sorted(batch, key = duration_in_frames, reverse=True)
	longest_sample = max(batch, key = duration_in_frames)[0]
	freq_size, max_seq_len = longest_sample.shape
	inputs = torch.zeros(len(batch), freq_size, max_seq_len, device = batch[0][0].device, dtype = batch[0][0].dtype)
	input_percentages = torch.FloatTensor(len(batch))
	target_sizes = torch.IntTensor(len(batch))
	targets, filenames = [], []
	for k, (tensor, target, filename) in enumerate(batch):
		seq_len = tensor.shape[1]
		inputs[k, :, :seq_len] = tensor
		input_percentages[k] = seq_len / float(max_seq_len)
		target_sizes[k] = len(target)
		targets.extend(target)
		filenames.append(filename)
	targets = torch.IntTensor(targets)
	return inputs, targets, filenames, input_percentages, target_sizes

def load_example(audio_path, transcript, sample_rate, window_size, window_stride, window, num_input_features, parse_transcript = lambda transcript: transcript, waveform_transform = None, feature_transform = None, normalize_features = True):
	signal, sample_rate = read_wav(audio_path, sample_rate = sample_rate)
	if waveform_transform is not None:
		signal, sample_rate = waveform_transform(signal, sample_rate); 
		#dirname = os.path.join('data', waveform_transform.__class__.__name__); os.makedirs(dirname, exist_ok = True); scipy.io.wavfile.write(os.path.join(dirname, f'{random.randint(0, int(1e9))}.wav'), sample_rate, signal.numpy())
		
	features = logfbank(signal, sample_rate, window_size, window_stride, window, num_input_features, normalize = normalize_features)
	if feature_transform is not None:
		features = feature_transform(features)

	transcript = parse_transcript(transcript)
	return features, transcript, audio_path

def logfbank_(signal, sample_rate, window_size, window_stride, window, num_input_features):
	window = getattr(scipy.signal, window)
	preemphasis = lambda signal, coeff: torch.cat([signal[:1], torch.sub(signal[1:], torch.mul(signal[:-1], coeff))])
	n_fft = int(sample_rate * (window_size + 1e-8))
	win_length = n_fft
	hop_length = int(sample_rate * (window_stride + 1e-8))
	signal = preemphasis(signal, coeff = 0.97)
	spect = librosa.stft(signal.numpy(), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True)
	spect = np.abs(spect) ** 2.0
	features = librosa.filters.mel(sample_rate, n_fft, n_mels=num_input_features, fmin=0, fmax=int(sample_rate/2)) @ spect
	features = torch.from_numpy(np.log(features + 1e-20))
	mean = features.mean(dim = 1, keepdim = True)
	std = features.std(dim = 1, keepdim = True)
	return (features - mean) / std

def logfbank(signal, sample_rate, window_size, window_stride, window, num_input_features, dither = 1e-5, eps = 1e-20, preemph = 0.97, normalize = True):
	preemphasis = lambda signal, coeff: torch.cat([signal[:1], signal[1:] - coeff * signal[:-1]])
	signal = signal / (signal.abs().max() + eps)
	signal = preemphasis(signal, coeff = preemph)
	win_length, hop_length = int(window_size * sample_rate), int(window_stride * sample_rate)
	n_fft = 2 ** math.ceil(math.log2(win_length))
	signal += dither * torch.randn_like(signal)
	window = getattr(torch, window)(win_length, periodic = False).type_as(signal)
	mel_basis = torch.from_numpy(librosa.filters.mel(sample_rate, n_fft, n_mels=num_input_features, fmin=0, fmax=int(sample_rate/2))).type_as(signal)
	power_spectrum = torch.stft(signal, n_fft, hop_length = hop_length, win_length = win_length, window = window, pad_mode = 'reflect', center = True).pow(2).sum(dim = -1)
	features = torch.log(torch.matmul(mel_basis, power_spectrum) + eps)
	if normalize:
		features = (features - features.mean(dim = 1, keepdim = True)) / (eps + features.std(dim = 1, keepdim = True))
	return features 

def read_wav(path, channel=-1, normalize = True, sample_rate = None, max_duration = None):
	sample_rate_, signal = scipy.io.wavfile.read(path)
	if len(signal.shape) > 1:
		if signal.shape[1] == 1:
			signal = signal.squeeze()
		elif channel == -1:
			signal = signal.mean(1)
		else:
			signal = signal[:, channel] 
		assert len(signal.shape) == 1

	if max_duration is not None:
		signal = signal[:int(max_duration * sample_rate_), ...]
	signal = torch.from_numpy(signal).to(torch.float32)
	if normalize:
		signal *= 1. / (signal.abs().max() + 1e-5)

	if sample_rate is not None and sample_rate_ != sample_rate:
		sample_rate_, signal = sample_rate, torch.from_numpy(librosa.resample(signal.numpy(), sample_rate_, sample_rate))
		#dirname = os.path.join('data', 'sample_ok_converted'); os.makedirs(dirname, exist_ok = True); scipy.io.wavfile.write(os.path.join(dirname, f'{random.randint(0, int(1e9))}.wav'), sample_rate, signal.numpy())

	return signal, sample_rate_
