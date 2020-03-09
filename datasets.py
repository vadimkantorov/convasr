#TODO: support forced mono even if transcript is given

import os
import re
import gzip
import time
import math
import json
import random
import functools
import itertools
import numpy as np
import torch.utils.data
import sentencepiece
import audio
import vad
import transcripts

class AudioTextDataset(torch.utils.data.Dataset):
	def __init__(self, source_paths, labels, sample_rate, frontend = None, waveform_transform_debug_dir = None, min_duration = None, max_duration = None, mono = True, delimiter = ',', segmented = False, vad_options = {}, time_padding_multiple = 1):
		self.labels = labels
		self.frontend = frontend
		self.sample_rate = sample_rate
		self.waveform_transform_debug_dir = waveform_transform_debug_dir
		self.vad_options = vad_options
		self.segmented = segmented
		self.time_padding_multiple = time_padding_multiple
		self.mono = mono

		gzopen = lambda data_path: open(data_path) if data_path.endswith('.json') else gzip.open(data_path, 'rt')
		duration = lambda example: sum(t.get('end', 0) - t.get('begin', 0) for t in example[-1]) if isinstance(example[-1], list) else len(example[-1])
		source_paths = source_paths if isinstance(source_paths, list) else [source_paths]
		
		if not self.segmented:
			self.examples = [[t] for data_path in source_paths for t in json.load(gzopen(data_path))]
		else:
			self.examples = [json.load(open(transcript_path) if os.path.exists(transcript_path) else [dict(audio_path = audio_path)]) for audio_path in source_paths for transcript_path in [audio_path + '.json']]
		
		self.examples = list(sorted(self.examples, key = duration))
		self.examples = list(filter(lambda example: (min_duration is None or min_duration <= duration(example)) and (max_duration is None or duration(example) <= max_duration), self.examples))

	def __getitem__(self, index):
		transcript = self.examples[index]
		audio_path = transcript[0]['audio_path']

		waveform_transform_debug = (lambda audio_path, sample_rate, signal: audio.write_audio(os.path.join(self.waveform_transform_debug_dir, os.path.basename(audio_path) + '.wav'), signal, sample_rate)) if self.waveform_transform_debug_dir else None
		
		if not self.segmented:
			transcript = transcript[0]
			signal, sample_rate = audio.read_audio(audio_path, sample_rate = self.sample_rate, mono = self.mono, dtype = torch.float32, normalize = True) if self.frontend is None or self.frontend.read_audio else (audio_path, self.sample_rate) 
			
			features = self.frontend(signal, waveform_transform_debug = waveform_transform_debug).squeeze(0) if self.frontend is not None else signal
			targets = [labels.encode(transcript['ref']) for labels in self.labels]
			ref_normalized, targets = zip(*targets)
		else:
			signal, sample_rate = audio.read_audio(audio_path, sample_rate = self.sample_rate, mono = self.mono, dtype = torch.int16, normalize = False)
			missing_transcript = not transcript or any(t.get('begin') is None and t.get('end') is None for t in transcript)
			if missing_transcript:
				assert len(signal) == 1
				ref_full = [self.labels[0].normalize_text(t['ref']) for t in transcript]
				speakers = [None] + list(sorted(set(t['speaker'] for t in transcript)))
				speaker = torch.cat([torch.full((len(ref) + 1, ), speakers.index(t['speaker']), dtype = torch.uint8).scatter_(0, torch.tensor(len(ref)), 0) for t, ref in zip(transcript, ref_full)])[:-1]
				transcript = [dict(channel = 0, begin = 0, end = signal.shape[1] / sample_rate, audio_path = audio_path, speaker = speaker, speakers = speakers, ref = ' '.join(ref_full))]
			
			transcript = [dict(dict(channel = 0, speaker = None), **t) for t in sorted(transcript, key = transcripts.sort_key)]
			features = [self.frontend(segment, waveform_transform_debug = waveform_transform_debug).squeeze(0) if self.frontend is not None else segment.unsqueeze(0) for t in transcript for segment in [signal[t['channel'], int(t['begin'] * sample_rate) : 1 + int(t['end'] * sample_rate)]]]
			targets = [[labels.encode(t.get('ref', ''))[1] for t in transcript] for labels in self.labels]

		return [transcript, features] + list(targets)
			
	def __len__(self):
		return len(self.examples)

	def collate_fn(self, batch):
		if self.segmented:
			batch = list(zip(*batch))

		meta, sample_x, *sample_y = batch[0]
		xmax_len, *ymax_len = [int(math.ceil(max(b[k].shape[-1] for b in batch) / self.time_padding_multiple)) * self.time_padding_multiple for k in range(1, len(batch[0]))]
		x = torch.zeros(len(batch), len(sample_x), xmax_len, dtype = sample_x.dtype)
		y = torch.zeros(len(batch), len(sample_y), max(ymax_len), dtype = torch.long)
		xlen, ylen = torch.zeros(len(batch), dtype = torch.float32), torch.zeros(len(batch), len(sample_y), dtype = torch.long)
		for k, (meta, sample_x, *sample_y) in enumerate(batch):
			xlen[k] = sample_x.shape[-1] / x.shape[-1]
			x[k, ..., :sample_x.shape[-1]] = sample_x
			for j, t in enumerate(sample_y):
				y[k, j, :t.shape[-1]] = t
				ylen[k, j] = len(t)
		
		return tuple(zip(*batch))[:1] + (x, xlen, y, ylen)

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
	repeat = '2'
	space = ' '
	blank = '|'
	unk = '*'
	word_start = '<'
	word_end = '>'
	candidate_sep = ';'
	
	space_sentencepiece = '\u2581'
	unk_sentencepiece = '<unk>'

	def __init__(self, lang, bpe = None, name = '', candidate_sep = ''):
		self.lang = lang
		self.name = name
		self.bpe = None
		if bpe:
			self.bpe = sentencepiece.SentencePieceProcessor()
			self.bpe.Load(bpe)
		
		self.alphabet = self.lang.ALPHABET
		self.blank_idx = len(self) - 1
		self.space_idx = self.blank_idx - 1
		self.repeat_idx = self.blank_idx - 2
		self.word_start_idx = self.alphabet.index(self.word_start) if self.word_start in self.alphabet else -1
		self.word_end_idx = self.alphabet.index(self.word_end) if self.word_end in self.alphabet else -1
		self.candidate_sep = candidate_sep
		self.chr2idx = {l: i for i, l in enumerate(str(self))}

	def split_candidates(self, text):
		return text.split(self.candidate_sep) if self.candidate_sep else [text]
	
	def normalize_text(self, text):
		return self.candidate_sep.join(self.lang.normalize_text(candidate) for candidate in self.split_candidates(text))# or self.unk 

	def encode(self, text):
		normalized = self.normalize_text(text)
		chars = self.split_candidates(normalized)[0]
		return normalized, torch.LongTensor([self.chr2idx[c] if i == 0 or c != chars[i - 1] else self.repeat_idx for i, c in enumerate(chars)] if self.bpe is None else self.bpe.EncodeAsIds(chars))

	def decode(self, idx : list, ts = None, I = None, speaker = None, channel = 0, speakers = None, replace_blank = True, replace_space = False, replace_repeat = True, key = 'hyp'):
		decode_ = lambda i, j: self.postprocess_transcript(''.join(self[idx[ij]] for ij in range(i, j + 1) if replace_repeat is False or ij == 0 or idx[ij] != idx[ij - 1]), replace_blank = replace_blank, replace_space = replace_space, replace_repeat = replace_repeat)
		speaker_ = lambda i, j: None if speaker is None else int(speaker[i:1 + j].max()) if speakers is None else speakers[int(speaker[i:1 + j].max())]
		channel_ = lambda i_, j_: channel if isinstance(channel, int) else int(channel[i_])

		if ts is None:
			return decode_(0, len(idx) - 1)

		pad = [self.space_idx] if replace_blank is False else [self.space_idx, self.blank_idx]
		
		transcript, i = [], None
		for j, k in enumerate(idx + [self.space_idx]):
			if k == self.space_idx and i is not None:
				while j == len(idx) or (j > 0 and idx[j] in pad):
					j -= 1
				
				i_, j_ = int(i if I is None else I[i]), int(j if I is None else I[j])
				transcript.append(dict(begin = float(ts[i_]), end = float(ts[j_]), i = i_, j = j_, channel = channel_(i_, j_), speaker = speaker_(i, j), **{key : decode_(i, j)}))

				i = None
			elif k not in [self.space_idx, self.blank_idx] and i is None:
				i = j
		return transcript

	def postprocess_transcript(self, word, replace_blank = True, replace_space = False, replace_repeat = True, replace_unk = True, collapse_repeat = False, phonetic_replace_groups = []):
		if replace_blank is not False:
			word = word.replace(self.blank, '' if replace_blank is True else replace_blank)
		if replace_unk is True:
			word = word.replace(self.unk, '' if replace_unk is True else replace_unk)
		if replace_space is not False:
			word = word.replace(self.space, replace_space)
		if replace_repeat is True:
			word = ''.join(c if i == 0 or c != self.repeat else word[i - 1] for i, c in enumerate(word))
		
		if collapse_repeat:
			word = ''.join(c if i == 0 or c != word[i - 1] else '' for i, c in enumerate(word))
		return word

	def __getitem__(self, idx):
		return {self.blank_idx : self.blank, self.repeat_idx : self.repeat, self.space_idx : self.space}.get(idx) or (self.alphabet[idx] if self.bpe is None else self.bpe.IdToPiece(idx).replace(self.space_sentencepiece, self.space).replace(self.unk_sentencepiece, self.unk))

	def __len__(self):
		return len(self.alphabet if self.bpe is None else self.bpe) + len([self.repeat, self.space, self.blank])

	def __str__(self):
		return self.alphabet + ''.join([self.repeat, self.space, self.blank])
