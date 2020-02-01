import os
import re
import gzip
import time
import math
import random
import subprocess
import numpy as np
import functools
import torch.utils.data
import scipy.io.wavfile
import librosa
import sentencepiece
import models
import itertools

class AudioTextDataset(torch.utils.data.Dataset):
	def __init__(self, source_paths, labels, sample_rate, frontend = None, waveform_transform_debug_dir = None, max_duration = None, delimiter = ','):
		self.labels = labels
		self.frontend = frontend
		self.sample_rate = sample_rate
		self.waveform_transform_debug_dir = waveform_transform_debug_dir
		self.examples = sum([list(sorted(((os.path.basename(data_or_path), row[0], row[1] if not row[1].endswith('.txt') else open(row[1]).read(), float(row[2]) if True and len(row) > 2 else -1) for line in (gzip.open(data_or_path, 'rt') if data_or_path.endswith('.gz') else open(data_or_path)) if '"' not in line for row in [line.split(delimiter)] if len(row) <= 2 or (max_duration is None or float(row[2]) < max_duration)), key = lambda t: t[-1])) for data_or_path in (source_paths if isinstance(source_paths, list) else [source_paths])], [])

	def __getitem__(self, index):
		dataset_name, audio_path, ref, duration = self.examples[index]
		signal, sample_rate = read_audio(audio_path, sample_rate = self.sample_rate) if self.frontend.read_audio else (audio_path, self.sample_rate) 
		# int16 or float?
		features = self.frontend(signal.unsqueeze(0), waveform_transform_debug = lambda audio_path, sample_rate, signal: write_wav(os.path.join(self.waveform_transform_debug_dir, os.path.basename(audio_path) + '.wav')) if self.waveform_transform_debug_dir else None).squeeze(0) if self.frontend is not None else signal
		ref_normalized = self.labels[0].encode(ref)[0]
		targets = [labels.encode(ref)[1] for labels in self.labels]
		return [dataset_name, audio_path, ref_normalized, features] + targets

	def __len__(self):
		return len(self.examples)

class BucketingBatchSampler(torch.utils.data.Sampler):
	def __init__(self, dataset, bucket, batch_size = 1, mixing = None):
		super().__init__(dataset)
		
		self.dataset = dataset
		self.batch_size = batch_size
		#self.mixing = mixing or ([1 / len(self.dataset.examples)] * len(self.dataset.examples))
		key = lambda example_idx: bucket(self.dataset.examples[example_idx])
		self.buckets = {k : list(g) for k, g in itertools.groupby(sorted(range(len(self.dataset)), key = key), key = key)}
		self.batch_idx = 0
		self.shuffle(epoch = 0)

	def __iter__(self):
		return iter(self.shuffled[self.batch_idx:])

	def __len__(self):
		return len(self.shuffled)

	def shuffle(self, epoch):
		rng = torch.Generator()
		rng.manual_seed(epoch)
		shuffle = lambda e: [e[k] for k in torch.randperm(len(e), generator = rng).tolist()]
		num_batches = int(math.ceil(len(self.dataset) / self.batch_size))
		batch_sequentially = lambda e: [e[i * self.batch_size : (1 + i) * self.batch_size] for i in range(int(math.ceil(len(e) / self.batch_size)))]
		batches = sum([batch_sequentially(shuffle(g)) for g in self.buckets.values()], [])
		#batches = batch_sequentially(list(range(len(self.dataset))))
		#mixing = [int(m * self.batch_size) for m in self.mixing]
		#inds = [chunk(i, num_batches) for k, subset in enumerate(self.dataset.examples) for i in [sum(map(len, self.dataset.examples[:k])) + torch.arange(len(subset))]]
		#batches = [torch.cat([i[torch.randperm(len(i), generator = generator)[:m]] for i, m in zip(t, mixing)]).tolist() for t in zip(*inds)]
		self.shuffled = [batches[k] for k in torch.randperm(len(batches), generator = rng).tolist()]

	def state_dict(self):
		return dict(batch_idx = self.batch_idx)

	def load_state_dict(self, state_dict):
		self.batch_idx = state_dict['batch_idx']

class Labels:
	blank = '|'
	space = ' '
	repeat = '2'
	word_start = '<'
	word_end = '>'
	candidate_sep = ';'

	def __init__(self, lang, bpe = None, name = ''):
		self.lang = lang
		self.name = name
		self.preprocess_text = lang.preprocess_text
		self.preprocess_word = lang.preprocess_word
		self.bpe = None
		if bpe:
			self.bpe = sentencepiece.SentencePieceProcessor()
			self.bpe.Load(bpe)
		self.alphabet = self.lang.LABELS.lower()# + self.lang.LABELS[:-1].upper()
		self.blank_idx = len(self) - 1
		self.space_idx = self.blank_idx - 1
		self.repeat_idx = self.blank_idx - 2
		self.word_start_idx = self.alphabet.index(self.word_start) if self.word_start in self.alphabet else -1
		self.word_end_idx = self.alphabet.index(self.word_end) if self.word_end in self.alphabet else -1

	def find_words(self, text):
		text = re.sub(r'([^\W\d]+)2', r'\1', text)
		text = self.preprocess_text(text)
		words = re.findall(r'-?\d+|-?\d+-\w+|\w+', text)
		return list(filter(bool, (''.join(c for c in self.preprocess_word(w) if c in self).strip() for w in words)))

	def normalize_text(self, text):
		return self.candidate_sep.join(' '.join(self.find_words(part)).lower().strip() for part in self.split_candidates(text)) or '*' 
		#return ' '.join(f'{w[:-1]}{w[-1].upper()}' for w in self.find_words(text.lower())) or '*' 

	def encode(self, text):
		normalized = self.normalize_text(text)
		chars = normalized.split(';')[0]
		chr2idx = {l: i for i, l in enumerate(str(self))}
		return normalized, torch.IntTensor([chr2idx[c] if i == 0 or c != chars[i - 1] else self.repeat_idx for i, c in enumerate(chars)] if self.bpe is None else self.bpe.EncodeAsIds(chars))

	def decode(self, idx : list, ts = None, I = None, replace_blank = True, replace_space = False, replace_repeat = True):
		decode_ = lambda i, j: self.postprocess_transcript2(''.join(self[idx[ij]] for ij in range(i, j + 1) if replace_repeat is False or ij == 0 or idx[ij] != idx[ij - 1]), replace_blank = replace_blank, replace_space = replace_space, replace_repeat = replace_repeat)
		
		if ts is None:
			return decode_(0, len(idx) - 1)

		words, i = [], None
		for j, k in enumerate(idx + [self.space_idx]):
			if k == self.space_idx and i is not None:
				j__ = j
				while j == len(idx) or (j > 0 and idx[j] in [self.space_idx, self.blank_idx]):
					j -= 1

				i_, j_ = int(i if I is None else I[i]), int(j if I is None else I[j])
				if j_ < i_:
					import IPython; IPython.embed()
				words.append(dict(word = decode_(i, j), begin = float(ts[i_]), end = float(ts[j_]), i = i_, j = j_))
				i = None
			elif k not in [self.space_idx, self.blank_idx] and i is None:
				i = j
		return words

	def split_candidates(self, text):
		return text.split(self.candidate_sep)

	#TODO: merge postprocess_transcript

	def postprocess_transcript2(self, word, replace_blank = True, replace_space = False, replace_repeat = True):
		if replace_blank is not False:
			word = word.replace(self.blank, '' if replace_blank is True else replace_blank)
		if replace_space is not False:
			word = word.replace(self.space, replace_space)
		if replace_repeat is True:
			word = ''.join(c if i == 0 or c != self.repeat else word[i - 1] for i, c in enumerate(word))
		return word

	def postprocess_transcript(self, text, phonetic_replace_groups = []):
		replaceblank = lambda s: s.replace(self.blank * 10, ' ').replace(self.blank, '')
		replace2 = lambda s: ''.join(c if i == 0 or c != self.repeat else s[i - 1] for i, c in enumerate(s))
		replace22 = lambda s: ''.join(c if i == 0 or c != s[i - 1] else '' for i, c in enumerate(s))
		replacestar = lambda s: s.replace('*', '')
		replacespace = lambda s, sentencepiece_space = '\u2581', sentencepiece_unk = '<unk>': s.replace(sentencepiece_space, ' ')
		replacecap = lambda s: ''.join(c + ' ' if c.isupper() else c for c in s)
		replacephonetic = lambda s: s.translate({ord(c) : g[0] for g in phonetic_replace_groups for c in g.lower()})
		replacepunkt = lambda s: s.replace(',', '').replace('.', '')

		return functools.reduce(lambda text, func: func(text), [replacepunkt, replacespace, replacecap, replaceblank, replace2, replace22, replacestar, replacephonetic, str.strip], text)

	def __getitem__(self, idx):
		return {self.blank_idx : self.blank, self.repeat_idx : self.repeat, self.space_idx : self.space}.get(idx) or (self.alphabet[idx] if self.bpe is None else self.bpe.IdToPiece(idx))

	def __len__(self):
		return len(self.alphabet if self.bpe is None else self.bpe) + len([self.repeat, self.space, self.blank])

	def __str__(self):
		return self.alphabet + ''.join([self.repeat, self.space, self.blank])
	
	def __contains__(self, chr):
		return chr.lower() in self.alphabet

class BatchCollater:
	def __init__(self, time_padding_multiple = 1):
		self.time_padding_multiple = time_padding_multiple

	def __call__(self, batch):
		dataset_name, audio_path, reference, sample_x, *sample_y = batch[0]
		xmax_len, ymax_len = [int(math.ceil(max(b[k].shape[-1] for b in batch) / self.time_padding_multiple)) * self.time_padding_multiple for k in [3, 4]]
		x = torch.zeros(len(batch), len(sample_x), xmax_len, dtype = torch.float32)
		y = torch.zeros(len(batch), len(sample_y), ymax_len, dtype = torch.long)
		xlen, ylen = torch.zeros(len(batch), dtype = torch.float32), torch.zeros(len(batch), len(sample_y), dtype = torch.long)
		for k, (dataset_name, audio_path, reference, input, *targets) in enumerate(batch):
			xlen[k] = input.shape[-1] / x.shape[-1]
			x[k, ..., :input.shape[-1]] = input
			for j, t in enumerate(targets):
				y[k, j, :t.shape[-1]] = t
				ylen[k, j] = len(t)
		dataset_name_, audio_path_, reference_, *_ = zip(*batch)
		#x: NCT, y: NLt, ylen: NL, xlen: N
		return dataset_name_, audio_path_, reference_, x, xlen, y, ylen

def read_audio(audio_path, sample_rate, normalize = True, mono = True, max_duration = None, dtype = torch.float32, byte_order = 'little'):
	if audio_path.endswith('.wav'):
		sample_rate_, signal = scipy.io.wavfile.read(audio_path) 
	else:
		num_channels = int(subprocess.check_output(['soxi', '-V0', '-c', audio_path]))
		sample_rate_, signal = sample_rate, torch.ShortTensor(torch.ShortStorage.from_buffer(subprocess.check_output(['sox', '-V0', audio_path, '-b', '16', '-e', 'signed', '--endian', byte_order, '-r', str(sample_rate), '-c', str(num_channels), '-t', 'raw', '-']), byte_order = byte_order)).reshape(-1, num_channels)

	signal = (signal if not mono else signal.squeeze(1) if signal.shape[1] == 1 else signal.float().mean(dim = 1)) if len(signal.shape) > 1 else (signal if mono else signal.unsqueeze(-1))
	if max_duration is not None:
		signal = signal[:int(max_duration * sample_rate_), ...]

	signal = torch.as_tensor(signal).to(dtype)
	if normalize:
		signal = models.normalize_signal(signal, dim = 0)
	if dtype is torch.float32 and sample_rate_ != sample_rate:
		signal, sample_rate_ = resample(signal, sample_rate_, sample_rate)

	assert sample_rate_ == sample_rate, 'Cannot resample non-float tensors because of librosa constraints'
	return signal, sample_rate_

def write_audio(audio_path, sample_rate, signal):
	assert audio_path.endswith('.wav')
	scipy.io.wavfile.write(audio_path, sample_rate, signal.numpy())

def resample(signal, sample_rate_, sample_rate):
	return torch.from_numpy(librosa.resample(signal.numpy(), sample_rate_, sample_rate)), sample_rate

def remove_silence(vad, signal, sample_rate, window_size):
	frame_len = int(window_size * sample_rate)
	voice = [False]
	voice.extend(vad.is_speech(signal[sample_idx : sample_idx + frame_len].numpy().tobytes(), sample_rate) if sample_idx + frame_len <= len(signal) else False for sample_idx in range(0, len(signal), frame_len))
	voice.append(False)
	voice = torch.tensor(voice)
	voice, _voice = voice[1:], voice[:-1]
	
	# 1. merge if gap < 1 sec
	# 2. remove if len < 0.5 sec
	# 3. filter by energy
	# 4. cut sends by energy

	
	begin_end = list(zip((~_voice & voice).nonzero().squeeze(1).tolist(), (~voice & _voice).nonzero().squeeze(1).tolist()))
	return ((frame_len * torch.IntTensor(begin_end)).float() / sample_rate).int().tolist()
