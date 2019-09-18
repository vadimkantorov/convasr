import os
import re
import csv
import gzip
import time
import math
import random
import functools
import torch.utils.data
import scipy.io.wavfile
import librosa
import sentencepiece
import models
import transforms

class AudioTextDataset(torch.utils.data.Dataset):
	def __init__(self, data_or_path, sample_rate, window_size, window_stride, window, num_input_features, labels, waveform_transform = None, feature_transform = None, max_duration = 20, normalize_features = True, waveform_transform_debug_dir = None):
		self.window_stride = window_stride
		self.window_size = window_size
		self.sample_rate = sample_rate
		self.window = window
		self.num_input_features = num_input_features
		self.labels = labels
		self.waveform_transform = waveform_transform
		self.feature_transform = feature_transform
		self.normalize_features = normalize_features
		self.waveform_transform_debug_dir = waveform_transform_debug_dir
		self.ids = [(row[0], row[1] if not row[1].endswith('.txt') else open(row[1]).read(), float(row[2]) if len(row) > 2 else -1) for row in csv.reader(gzip.open(data_or_path, 'rt') if data_or_path.endswith('.gz') else open(data_or_path)) if len(row) <= 2 or float(row[2]) < max_duration] if isinstance(data_or_path, str) else [d for d in data_or_path if d[-1] == -1 or d[-1] < max_duration]

	def __getitem__(self, index):
		audio_path, transcript, duration = self.ids[index]

		signal, sample_rate = (audio_path, self.sample_rate) if isinstance(self.waveform_transform, transforms.SoxAug) else read_wav(audio_path, sample_rate = self.sample_rate)
		if self.waveform_transform is not None:
			signal, sample_rate = self.waveform_transform(signal, self.sample_rate)
		
		if self.waveform_transform_debug_dir:
			scipy.io.wavfile.write(os.path.join(self.waveform_transform_debug_dir, os.path.basename(audio_path)), self.sample_rate, signal.numpy())

		#features = signal
		features = models.logfbank(signal, self.sample_rate, self.window_size, self.window_stride, self.window, self.num_input_features, normalize = self.normalize_features)
		if self.feature_transform is not None:
			features, sample_rate = self.feature_transform(features, self.sample_rate)

		transcript = self.labels.parse(transcript)
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
		self.shuffled = False

	def __iter__(self):
		for ids in self.bins[self.batch_idx:]:
			random.shuffle(ids)
			yield ids
		self.batch_idx = 0

	def __len__(self):
		return len(self.bins)

	def shuffle(self, epoch):
		if not self.shuffled:
			random.shuffle(self.bins)
			self.batch_idx = 0
		self.shuffled = False

	def state_dict(self, batch_idx):
		return dict(bins = self.bins, batch_idx = batch_idx)

	def load_state_dict(self, state_dict):
		self.bins = state_dict['bins']
		self.batch_idx = state_dict['batch_idx']
		self.shuffled = True

replace2 = lambda s: ''.join(c if i == 0 or c != '2' else s[i - 1] for i, c in enumerate(s))
replace22 = lambda s: ''.join(c if i == 0 or c != s[i - 1] else '' for i, c in enumerate(s))
replacestar = lambda s: s.replace('*', '')
replacespace = lambda s: s.replace('<', ' ').replace('>', ' ')

class Labels:
	blank = '|'
	space = ' '
	word_start = '<'
	word_end = '>'
	repeat = '2'

	def __init__(self, lang, bpe = None):
		self.preprocess_text = lang.preprocess_text
		self.preprocess_word = lang.preprocess_word
		self.bpe = None
		if bpe:
			self.bpe = sentencepiece.SentencePieceProcessor()
			self.bpe.Load(bpe)
		self.idx2chr = lang.LABELS
		self.chr2idx = {l: i for i, l in enumerate(self.idx2chr + self.repeat + self.space + self.blank)}
		self.blank_idx = len(self) - 1
		self.space_idx = self.blank_idx - 1
		self.repeat_idx = self.blank_idx - 2

	def find_words(self, text):
		text = re.sub(r'([^\W\d]+)2', r'\1', text)
		text = self.preprocess_text(text)
		words = re.findall(r'-?\d+|-?\d+-\w+|\w+', text)
		return list(filter(bool, (''.join([c for c in self.preprocess_word(w) if c in self]).strip() for w in words)))

	def normalize_text(self, text):
		return ' '.join(f'{w}' for w in self.find_words(text)).upper().strip() or '*' 
		#return ''.join(f'<{w}>' for w in self.find_words(text)).upper().strip() or '*' 

	def parse(self, text):
		chars = self.normalize_text(text)
		return [self.chr2idx[chr] if i == 0 or chr != chars[i - 1] else self.repeat_idx for i, chr in enumerate(chars)] if self.bpe is None else self.bpe.EncodeAsIds(chars)

	def normalize_transcript(self, text):
		return functools.reduce(lambda text, func: func(text), [replacespace, replace2, replace22, replacestar, str.strip], text)

	def idx2str(self, idx):
		i2s = lambda i: '' if len(i) == 0 else self.normalize_transcript(''.join(map(self.__getitem__, i)) if self.bpe is None else self.bpe.DecodeIds(i)) if not isinstance(i[0], list) else list(map(i2s, i))
		return i2s(idx)

	def __getitem__(self, idx):
		return self.idx2chr[idx] if self.bpe is None else self.bpe.IdToPiece(idx)

	def __len__(self):
		return (len(self.idx2chr) if self.bpe is None else len(self.bpe)) + len([self.repeat, self.space, self.blank])

	def __str__(self):
		return self.idx2chr
	
	def __contains__(self, chr):
		return chr.upper() in self.idx2chr

def unpack_targets(packed_targets, target_lengths):
	targets, offset = [], 0
	for length in target_lengths:
		targets.append(packed_targets[offset:offset + length])
		offset += length
	return targets

def collate_fn(batch, pad_to = 16):
	batch = list(sorted(batch, key = lambda example: example[0].shape[-1], reverse = True))
	max_seq_len = (1 + (batch[0][0].shape[-1] // pad_to)) * pad_to
	inputs = torch.zeros(len(batch), *(batch[0][0].shape[:-1] + (max_seq_len, )), device = batch[0][0].device, dtype = batch[0][0].dtype)
	input_percentages = torch.FloatTensor(len(batch))
	target_lengths = torch.IntTensor(len(batch))
	packed_targets, filenames = [], []
	for k, (input, target, filename) in enumerate(batch):
		inputs[k, ..., :input.shape[-1]] = input
		input_percentages[k] = input.shape[-1] / float(max_seq_len)
		target_lengths[k] = len(target)
		packed_targets.extend(target)
		filenames.append(filename)
	return inputs, torch.IntTensor(packed_targets), filenames, input_percentages, target_lengths

def read_wav(path, normalize = True, sample_rate = None, max_duration = None):
	sample_rate_, signal = scipy.io.wavfile.read(path)
	signal = (signal.squeeze() if signal.shape[1] == 1 else signal.mean(1)) if len(signal.shape) > 1 else signal
	if max_duration is not None:
		signal = signal[:int(max_duration * sample_rate_), ...]

	signal = torch.from_numpy(signal).to(torch.float32)
	if normalize:
		signal = models.normalize_signal(signal)
	if sample_rate is not None and sample_rate_ != sample_rate:
		sample_rate_, signal = resample(signal, sample_rate_, sample_rate)

	return signal, sample_rate_

def resample(signal, sample_rate_, sample_rate):
	return sample_rate, torch.from_numpy(librosa.resample(signal.numpy(), sample_rate_, sample_rate))
