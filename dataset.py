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
	def __init__(self, source_paths, sample_rate, window_size, window_stride, window, num_input_features, labels, parse_transcript = True, waveform_transform = None, feature_transform = None, max_duration = None, normalize_features = True, waveform_transform_debug_dir = None):
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
		self.parse_transcript = parse_transcript

		self.ids = [list(sorted(((os.path.basename(data_or_path), row[0], row[1] if not row[1].endswith('.txt') else open(row[1]).read(), float(row[2]) if True and len(row) > 2 else -1) for row in csv.reader(gzip.open(data_or_path, 'rt') if data_or_path.endswith('.gz') else open(data_or_path), delimiter=',') if len(row) <= 2 or (max_duration is not None and True and float(row[2]) < max_duration)), key = lambda t: t[-1])) for data_or_path in (source_paths if isinstance(source_paths, list) else [source_paths])]

	def __getitem__(self, index):
		for ids in self.ids:
			if index < len(ids):
				dataset_name, audio_path, transcript, duration = ids[index]
				break
			else:
				index -= len(ids)

		signal, sample_rate = (audio_path, self.sample_rate) if isinstance(self.waveform_transform, transforms.SoxAug) else read_wav(audio_path, sample_rate = self.sample_rate)
		if self.waveform_transform is not None:
			signal, sample_rate = self.waveform_transform(signal, self.sample_rate, dataset_name = dataset_name)
		
		if self.waveform_transform_debug_dir:
			scipy.io.wavfile.write(os.path.join(self.waveform_transform_debug_dir, os.path.basename(audio_path)), self.sample_rate, signal.numpy())

		#features = signal
		features = models.logfbank(signal, self.sample_rate, self.window_size, self.window_stride, self.window, self.num_input_features, normalize = self.normalize_features)
		if self.feature_transform is not None:
			features, sample_rate = self.feature_transform(features, self.sample_rate, dataset_name = dataset_name)

		transcript = self.labels.parse(transcript, idx = self.parse_transcript)
		return features, transcript, audio_path, dataset_name

	def __len__(self):
		return sum(map(len, self.ids))

class BucketingSampler(torch.utils.data.Sampler):
	def __init__(self, dataset, batch_size = 1, mixing = None):
		super().__init__(dataset)
		self.dataset = dataset
		self.batch_size = batch_size
		self.mixing = mixing or ([1 / len(self.dataset.ids)] * len(self.dataset.ids))
		self.shuffle(epoch = 0)

	def __iter__(self):
		for batch in self.shuffled[self.batch_idx:]:
			yield batch
			self.batch_idx += 1

	def __len__(self):
		return len(self.shuffled)

	def shuffle(self, epoch, batch_idx = 0):
		self.epoch = epoch
		self.batch_idx = batch_idx
		generator = torch.Generator()
		generator.manual_seed(self.epoch)

		mixing = [int(m * self.batch_size) for m in self.mixing]
		chunk = lambda xs, chunks: [xs[i * batch_size : (1 + i) * batch_size] for batch_size in [ len(xs) // chunks ] for i in range(chunks)]
		num_batches = int(len(self.dataset.ids[0]) // self.batch_size + 0.5)
		inds = [chunk(i, num_batches) for k, subset in enumerate(self.dataset.ids) for i in [sum(map(len, self.dataset.ids[:k])) + torch.arange(len(subset))]]
		batches = [torch.cat([i[torch.randperm(len(i), generator = generator)[:m]] for i, m in zip(t, mixing)]).tolist() for t in zip(*inds)]
		self.shuffled = [batches[k] for k in torch.randperm(len(batches), generator = generator).tolist()]

	def state_dict(self, batch_idx):
		return dict(epoch = self.epoch, batch_idx = batch_idx, shuffled = self.shuffled)

	def load_state_dict(self, state_dict):
		self.epoch, self.batch_idx, self.shuffled = state_dict['epoch'], state_dict['batch_idx'], (state_dict.get('shuffled') or self.shuffled)

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
		self.alphabet = lang.LABELS.lower()
		self.blank_idx = len(self) - 1
		self.space_idx = self.blank_idx - 1
		self.repeat_idx = self.blank_idx - 2

	def find_words(self, text):
		text = re.sub(r'([^\W\d]+)2', r'\1', text)
		text = self.preprocess_text(text)
		words = re.findall(r'-?\d+|-?\d+-\w+|\w+', text)
		return list(filter(bool, (''.join(c for c in self.preprocess_word(w) if c in self).strip() for w in words)))

	def normalize_text(self, text):
		return ';'.join(' '.join(self.find_words(part)).lower().strip() for part in text.split(';')) or '*' 
		#return ''.join(f'<{w}>' for w in self.find_words(text)).upper().strip() or '*' 

	def parse(self, text, idx = True):
		chars = self.normalize_text(text)
		if not idx:
			return chars

		chr2idx = {l: i for i, l in enumerate(self.alphabet + self.repeat + self.space + self.blank)}
		return torch.IntTensor([chr2idx[c] if i == 0 or c != chars[i - 1] else self.repeat_idx for i, c in enumerate(chars)] if self.bpe is None else self.bpe.EncodeAsIds(chars))

	def normalize_transcript(self, text):
		return functools.reduce(lambda text, func: func(text), [replacespace, replace2, replace22, replacestar, str.strip], text)

	def idx2str(self, idx, lengths = None, blank = None, repeat = None):
		i2s_ = lambda i: '' if len(i) == 0 else self.normalize_transcript(''.join(map(self.__getitem__, i)) if self.bpe is None else self.bpe.DecodeIds(i)) if not blank else ''.join(blank if idx == self.blank_idx else self[idx] if k == 0 or idx == self.space_idx or idx != i[k - 1] else (repeat if repeat is not None else self[idx]) for k, idx in enumerate(i))
		i2s = lambda i: i2s_(i) if len(i) == 0 or not isinstance(i[0], list) else list(map(i2s, i))
		if torch.is_tensor(idx):
			idx = idx.tolist()
		idx = idx if lengths is None else [i[:l] for i, l in zip(idx, lengths)]
		return i2s(idx)

	def __getitem__(self, idx):
		return {self.blank_idx : self.blank, self.repeat_idx : self.repeat, self.space_idx : self.space}.get(idx) or (self.alphabet[idx] if self.bpe is None else self.bpe.IdToPiece(idx))

	def __len__(self):
		return (len(self.alphabet) if self.bpe is None else len(self.bpe)) + len([self.repeat, self.space, self.blank])

	def __str__(self):
		return self.alphabet + ''.join([self.repeat, self.space, self.blank])
	
	def __contains__(self, chr):
		return chr.lower() in self.alphabet

def collate_fn(batch, pad_to = 128):
	sample_inputs, sample_targets, *_ = batch[0]
	inputs_max_len, targets_max_len = [(1 + max( (b[k].shape[-1] if torch.is_tensor(b[k]) else len(b[k]))  for b in batch) // pad_to) * pad_to for k in [0, 1]]
	inputs = sample_inputs.new_zeros(len(batch), *(sample_inputs.shape[:-1] + (inputs_max_len,)))
	targets = sample_targets.new_zeros(len(batch), *(sample_targets.shape[:-1] + (targets_max_len,))) if torch.is_tensor(sample_targets) else []
	input_percentages, target_lengths, audio_paths, dataset_names = [], [], [], []
	for k, (input, target, audio_path, dataset_name) in enumerate(batch):
		inputs[k, ..., :input.shape[-1]] = input
		if torch.is_tensor(target):
			targets[k, ..., :target.shape[-1]] = target
		else:
			targets.append(target)
		input_percentages.append(input.shape[-1] / float(inputs_max_len))
		target_lengths.append(len(target))
		audio_paths.append(audio_path)
		dataset_names.append(dataset_name)
	return inputs, targets, torch.FloatTensor(input_percentages), torch.IntTensor(target_lengths), audio_paths, dataset_names

def read_wav(path, normalize = True, stereo = False, sample_rate = None, max_duration = None):
	sample_rate_, signal = scipy.io.wavfile.read(path)
	signal = (signal if stereo else signal.squeeze(1) if signal.shape[1] == 1 else signal.mean(1)) if len(signal.shape) > 1 else (signal if not stereo else signal[..., None])
	if max_duration is not None:
		signal = signal[:int(max_duration * sample_rate_), ...]

	signal = torch.from_numpy(signal).to(torch.float32)
	if normalize:
		signal = models.normalize_signal(signal, dim = 0)
	if sample_rate is not None and sample_rate_ != sample_rate:
		sample_rate_, signal = resample(signal, sample_rate_, sample_rate)

	return signal, sample_rate_

def resample(signal, sample_rate_, sample_rate):
	return sample_rate, torch.from_numpy(librosa.resample(signal.numpy(), sample_rate_, sample_rate))
