import os
import re
import csv
import gzip
import random
import numpy as np
import torch.utils.data
import scipy.io.wavfile
import scipy.signal
import librosa

class SpectrogramDataset(torch.utils.data.Dataset):
	def __init__(self, data_or_path, sample_rate, window_size, window_stride, window, num_input_features, labels, transform = lambda x: x, max_duration = 20, noise_level = None, noise_data_path = None):
		self.window_stride = window_stride
		self.window_size = window_size
		self.sample_rate = sample_rate
		self.window = getattr(scipy.signal, window)
		self.num_input_features = num_input_features
		self.labels = labels
		self.transform = transform
		self.ids = [(row[0], row[1], float(row[2]) if len(row) > 2 else -1) for row in csv.reader(gzip.open(data_or_path, 'rt') if data_or_path.endswith('.gz') else open(data_or_path)) if len(row) <= 2 or float(row[2]) < max_duration] if isinstance(data_or_path, str) else [d for d in data_or_path if d[-1] == -1 or d[-1] < max_duration]
		self.noise_level = [noise_level] if isinstance(noise_level, float) else noise_level
		self.noise_paths = list(map(str.strip, open(noise_data_path))) if noise_data_path is not None else []

	def __getitem__(self, index):
		audio_path, transcript, duration = self.ids[index]
		noise_path, noise_level = None, None
		if self.noise_paths:
			noise_path = self.noise_paths[hash(audio_path) % len(self.noise_paths)]
			noise_level = self.noise_level[0] if len(self.noise_level) == 1 else random.uniform(*self.noise_level)
		features, transcript, audio_path = load_example(audio_path, transcript, self.sample_rate, self.window_size, self.window_stride, self.window, self.num_input_features, self.labels.parse, noise_path = noise_path, noise_level = noise_level)
		features = self.transform(features) if self.transform is not None else features
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
	epsilon = '|'

	def __init__(self, char_labels, preprocess_text = lambda text: text, preprocess_word = lambda word: word):
		self.char_labels = char_labels
		self.labels_map = {l: i for i, l in enumerate(char_labels)}
		self.preprocess_text = preprocess_text
		self.preprocess_word = preprocess_word

	def find_words(self, text):
		text = re.sub(r'([^\W\d]+)2', r'\1', text)
		text = self.preprocess_text(text)
		words = re.findall(r'-?\d+|-?\d+-\w+|\w+', text)
		return list(filter(bool, (''.join([c for c in self.preprocess_word(w) if c.upper() in self.labels_map]).strip() for w in words)))

	def parse(self, text):
		if text.startswith('!clean:'):
			return [self.labels_map[x] for x in text.replace('!clean:', '', 1).strip()]
		chars = ' '.join(self.find_words(text)).upper().strip() or '*'
		return [self.labels_map[c] if i == 0 or c != chars[i - 1] else self.labels_map['2'] for i, c in enumerate(chars)]

	def render_transcript(self, codes):
		return ''.join([self.char_labels[i] for i in codes])

	def chr2idx(self, chr):
		return self.char_labels.index(chr)

	def idx2chr(self, idx):
		return self.char_labels[idx]

def get_cer_wer(decoder, transcript, reference):
	reference = reference.strip()
	transcript = transcript.strip()
	wer_ref = float(len(reference.split()) or 1)
	cer_ref = float(len(reference.replace(' ','')) or 1)
	if reference == transcript:
		return 0, 0, wer_ref, cer_ref
	else:
		wer = decoder.wer(transcript, reference)
		cer = decoder.cer(transcript, reference)
	return wer, cer, wer_ref, cer_ref

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
	inputs = torch.zeros(len(batch), freq_size, max_seq_len)
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

def mix_noise(signal, noise, noise_level):
	noise = torch.cat([noise] * (1 + len(signal) // len(noise)))[:len(signal)]
	return signal + noise * noise_level

def load_example(audio_path, transcript, sample_rate, window_size, window_stride, window, num_input_features, parse_transcript = lambda transcript: transcript, noise_path = None, noise_level = None):
	signal, sample_rate_ = read_wav(audio_path, sample_rate = sample_rate)
	if noise_path is not None and noise_level is not None:
		noise, sample_rate_ = read_wav(noise_path, sample_rate = sample_rate)
		signal = mix_noise(signal, noise, noise_level)
		
	features = logfbanknorm(signal, sample_rate, window_size, window_stride, window, num_input_features)
	transcript = parse_transcript(transcript)
	return features, transcript, audio_path

def logfbanknorm(signal, sample_rate, window_size, window_stride, window, num_input_features):
	preemphasis = lambda signal, coeff: torch.cat([signal[:1], torch.sub(signal[1:], torch.mul(signal[:-1], coeff))])
	n_fft = int(sample_rate * (window_size + 1e-8))
	win_length = n_fft
	hop_length = int(sample_rate * (window_stride + 1e-8))
	signal = preemphasis(signal, coeff = 0.97)
	spect = librosa.stft(signal.numpy(), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True)
	spect = np.abs(spect) ** 2.0
	features = librosa.filters.mel(sample_rate, n_fft, n_mels=num_input_features, fmin=0, fmax=int(sample_rate/2)) @ spect
	features = np.log(features + 1e-20)
	mean = np.mean(features, axis = 1, keepdims = True)
	std_dev = np.std(features, axis = 1, keepdims = True)
	features = (features - mean) / std_dev
	return torch.from_numpy(features)

def read_wav(path, channel=-1, normalize = True, sample_rate = None):
	sample_rate_, signal = scipy.io.wavfile.read(path)
	signal = torch.from_numpy(signal).to(torch.float32)
	if normalize:
		signal *= 1. / (signal.abs().max() + 1e-5)

	if sample_rate is not None and sample_rate_ != sample_rate:
		sample_rate_, signal = sample_rate, torch.from_numpy(librosa.resample(signal.numpy(), sample_rate_, sample_rate))

	if len(signal.shape) > 1:
		if signal.shape[1] == 1:
			signal = signal.squeeze()
		elif channel == -1:
			signal = signal.mean(1)
		else:
			signal = signal[:, channel] 
		assert len(signal.shape) == 1
	return signal, sample_rate_
