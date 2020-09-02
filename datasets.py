#TODO: support forced mono even if transcript is given

import os
import gzip
import math
import json
import itertools
import functools
import importlib
import torch.utils.data
import sentencepiece
import audio
import utils
import transcripts

def worker_init_fn(worker_id, num_threads = 1):
	utils.set_random_seed(worker_id)
	utils.reset_cpu_threads(num_threads)

class AudioTextDataset(torch.utils.data.Dataset):
	def __init__(
		self,
		data_paths,
		labels,
		sample_rate,
		frontend = None,
		speakers = None,
		waveform_transform_debug_dir = None,
		min_duration = None,
		max_duration = None,
		duration_filter = True,
		min_ref_len = None,
		max_ref_len = None,
		ref_len_filter = True,
		mono = True,
		segmented = False,
		time_padding_multiple = 1,
		audio_backend = None,
		exclude = set(),
		join_transcript = False
	):
		self.join_transcript = join_transcript
		self.max_duration = max_duration
		self.labels = labels
		self.frontend = frontend
		self.sample_rate = sample_rate
		self.waveform_transform_debug_dir = waveform_transform_debug_dir
		self.segmented = segmented
		self.time_padding_multiple = time_padding_multiple
		self.mono = mono
		self.audio_backend = audio_backend
		self.speakers = speakers

		maybegzopen = lambda data_path: gzip.open(data_path, 'rt') if data_path.endswith('.gz') else open(data_path)
		duration = lambda example: sum(map(transcripts.compute_duration, example))
		data_paths = data_paths if isinstance(data_paths, list) else [data_paths]

		def read_transcript(data_path):
			transcript_path = data_path + '.json' if '.json' not in data_path else data_path
			return json.load(maybegzopen(transcript_path)
								) if any(map(transcript_path.endswith, ['.json', '.json.gz'])) else [dict(audio_path = data_path)]

		self.examples = [
			list(g) for data_path in data_paths for k,
			g in itertools
			.groupby(sorted(read_transcript(data_path), key = transcripts.sort_key), key = transcripts.group_key)
		]
		self.examples = sorted(self.examples, key = duration)
		self.examples = list(
			filter(
				lambda example: (min_duration is None or min_duration <= duration(example)) and
				(max_duration is None or duration(example) <= max_duration),
				self.examples
			)
		) if duration_filter else self.examples

		if len(exclude) > 0:
			self.examples = [e for e in self.examples if transcripts.audio_name(e[0]) not in exclude]
		'''
		# TODO: don't forget to add this to transcribe.py!
		def safe_coding_for_audio_lenghts:
			duration = max(transcripts.compute_duration(t, hours=True) for t in meta)
			if x.numel() == 0 or (args.skip_file_longer_than_hours and duration > args.skip_file_longer_than_hours):
				print(
						f'Skipping [{audio_path}]. Size: {x.numel()}, duration: {duration} hours (>{args.skip_file_longer_than_hours})')
				continue
		'''

	def __getitem__(self, index):
		transcript = self.examples[index]
		audio_path = transcript[0]['audio_path']

		waveform_transform_debug = (
			lambda audio_path,
			sample_rate,
			signal: audio.write_audio(
				os.path.join(self.waveform_transform_debug_dir, os.path.basename(audio_path) + '.wav'),
				signal,
				sample_rate
			)
		) if self.waveform_transform_debug_dir else None

		if not self.segmented:
			transcript = transcript[0]
			signal, sample_rate = audio.read_audio(audio_path, sample_rate = self.sample_rate, mono = self.mono, backend = self.audio_backend, duration=self.max_duration) if self.frontend is None or self.frontend.read_audio else (audio_path, self.sample_rate)

			transcript = dict(dict(audio_name = os.path.basename(transcript['audio_path'])), **transcript)
			features = self.frontend(signal, waveform_transform_debug = waveform_transform_debug
										).squeeze(0) if self.frontend is not None else signal
			targets = [labels.encode(transcript['ref']) for labels in self.labels]
			ref_normalized, targets = zip(*targets)
		else:
			signal, sample_rate = audio.read_audio(audio_path, sample_rate = self.sample_rate, mono = self.mono, backend = self.audio_backend, duration=self.max_duration)
			replace_transcript = self.join_transcript or \
                               not transcript or \
                               (any(t.get('begin') is None and t.get('end') is None for t in transcript) and \
                                all(t.get('ref') is not None for t in transcript))
			normalize_text = True

			if replace_transcript:
				assert len(signal) == 1
				ref_full = [self.labels[0].normalize_text(t['ref']) for t in transcript]
				speakers = [None] + list(sorted(set(t['speaker'] for t in transcript if t.get('speaker') is not None)))
				speaker = torch.cat([
					torch.full((len(ref) + 1, ), speakers.index(t.get('speaker')),
								dtype = torch.uint8).scatter_(0, torch.tensor(len(ref)), 0) for t,
					ref in zip(transcript, ref_full)
				])[:-1]
				transcript = [dict(speaker = speaker, speakers = speakers, ref = ' '.join(ref_full))]
				normalize_text = False

			transcript = [
				dict(
					dict(
						audio_path = audio_path,
						channel = channel,
						speaker = self.speakers[channel] if self.speakers else None,
						begin = 0,
						end = signal.shape[1] / sample_rate
					),
					**t
				)
				for t in sorted(transcript, key = transcripts.sort_key)
				for channel in ([t['channel']] if 'channel' in t else range(len(signal)))
			]
			features = [
				self.frontend(segment, waveform_transform_debug = waveform_transform_debug).squeeze(0)
				if self.frontend is not None else segment.unsqueeze(0)
				for t in transcript
				for segment in [signal[t['channel'], int(t['begin'] * sample_rate):1 + int(t['end'] * sample_rate)]]
			]
			targets = [[labels.encode(t.get('ref', ''), normalize = normalize_text)[1]
						for t in transcript]
						for labels in self.labels]

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
			xlen[k] = sample_x.shape[-1] / x.shape[-1] if x.shape[-1] > 0 else 1.0
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
		self.buckets = {
			k: list(g)
			for k, g in itertools.groupby(sorted(range(len(self.dataset)), key = key), key = key)
		}
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
		batch_sequentially = lambda e: [
			e[i * self.batch_size:(1 + i) * self.batch_size] for i in range(int(math.ceil(len(e) / self.batch_size)))
		]
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

	def __init__(self, lang, bpe = None, name = '', candidate_sep = '', normalize_text_config = {}):
		self.name = name
		self.bpe = None
		if bpe:
			self.bpe = sentencepiece.SentencePieceProcessor()
			self.bpe.Load(bpe)

		self.alphabet = lang.ALPHABET
		self.lang_normalize_text = lang.normalize_text
		self.lang_stem = lang.stem
		self.blank_idx = len(self) - 1
		self.space_idx = self.blank_idx - 1
		self.repeat_idx = self.blank_idx - 2
		self.word_start_idx = self.alphabet.index(self.word_start) if self.word_start in self.alphabet else -1
		self.word_end_idx = self.alphabet.index(self.word_end) if self.word_end in self.alphabet else -1
		self.candidate_sep = candidate_sep
		self.chr2idx = {l: i for i, l in enumerate(str(self))}
		self.normalize_text_config = normalize_text_config

	def split_candidates(self, text):
		return text.split(self.candidate_sep) if self.candidate_sep else [text]

	def normalize_word(self, word):
		return word

		#TODO: use https://www.oreilly.com/library/view/python-cookbook/0596001673/ch03s15.html
		#_w_ = lambda w: self.space + word + self.space
		#def replace_stem(acc, u, v):
		#	stem, inflection = self.lang.stem(acc, inflection = True)
		#	return stem.replace(self.space + u, v) + invlection
		#
		#word = _w_(word)
		#word = functools.reduce(lambda acc, uv: acc.replace(*uv), [(_w_(u), _w_(v)) for u, v in self.replace_full_forms.items()], word)
		#word = functools.reduce(lambda acc, uv: acc.replace(*uv), self.replace_subwords_forms.items(), word)
		#word = functools.reduce(lambda acc, uv: acc.replace(*uv), [(_w_(u), self.unk) for u, v in self.replace_full_forms_by_unk], word)
		#word = functools.reduce(lambda acc, uv: replace(acc, *uv), self.replace_stems.items(), word)
		#word = word.translate({c : None for c in self.remove_chars})
		#return word.strip()

	def normalize_text(self, text):
		return self.candidate_sep.join(
			self.space.join(map(self.normalize_word, self.lang_normalize_text(candidate).split(self.space))) for candidate in self.split_candidates(text)
		)  # or self.unk

	def encode(self, text, normalize = True):
		normalized = self.normalize_text(text) if normalize else text
		chars = self.split_candidates(normalized)[0]
		return normalized, torch.LongTensor([self.chr2idx[c] if i == 0 or c != chars[i - 1] else self.repeat_idx for i, c in enumerate(chars)] if self.bpe is None else self.bpe.EncodeAsIds(chars))

	def decode(
		self,
		idx: list,
		ts = None,
		I = None,
		speaker = None,
		channel = 0,
		speakers = None,
		replace_blank = True,
		replace_blank_series = False,
		replace_space = False,
		replace_repeat = True,
		key = 'hyp'
	):
		decode_ = lambda i, j: self.postprocess_transcript(''.join(self[idx[k]] for k in range(i, j + 1) if replace_repeat is False or k == 0 or idx[k] != idx[k - 1]), replace_blank = replace_blank, replace_space = replace_space, replace_repeat = replace_repeat)
		speaker_ = lambda i, j: (int(speaker[i:1 + j].max()) if torch.is_tensor(speaker) else speaker) if speaker is not None and speakers is None else speakers[int(speaker[i:1 + j].max())] if speaker is not None and speakers is not None else None
		channel_ = lambda i_, j_: channel if isinstance(channel, int) else int(channel[i_])

		idx = torch.as_tensor(idx).tolist()
		if ts is None:
			return decode_(0, len(idx) - 1)

		if replace_blank_series:
			blanks = ''.join(self.blank if i == self.blank_idx else '_' for i in idx)
			blanks = blanks.replace(self.blank * replace_blank_series, self.space * replace_blank_series)
			for i, c in enumerate(blanks):
				if c == self.space:
					idx[i] = self.space_idx

		silence = [self.space_idx] if replace_blank is False else [self.space_idx, self.blank_idx]

		transcript, i = [], None
		for j, k in enumerate(idx + [self.space_idx]):
			if k == self.space_idx and i is not None:
				while j == len(idx) or (j > 0 and idx[j] in silence):
					j -= 1

				i_, j_ = int(i if I is None else I[i]), int(j if I is None else I[j])
				transcript.append(
					dict(
						begin = float(ts[i_]),
						end = float(ts[j_]),
						i = i_,
						j = j_,
						channel = channel_(i_, j_),
						speaker = speaker_(i, j),
						**{key: decode_(i, j)}
					)
				)

				i = None
			elif k not in silence and i is None:
				i = j
		return transcript

	def postprocess_transcript(
		self,
		text,
		replace_blank = True,
		replace_space = False,
		replace_repeat = True,
		replace_unk = True,
		collapse_repeat = False,
		strip = True,
		phonetic_replace_groups = []
	):
		if strip:
			text = text.strip()
		if replace_blank is not False:
			text = text.replace(self.blank, '' if replace_blank is True else replace_blank)
		if replace_unk is True:
			text = text.replace(self.unk, '' if replace_unk is True else replace_unk)
		if replace_space is not False:
			text = text.replace(self.space, replace_space)
		if replace_repeat is True:
			text = ''.join(c if i == 0 or c != self.repeat else text[i - 1] for i, c in enumerate(text))
		if collapse_repeat:
			text = ''.join(c if i == 0 or c != text[i - 1] else '' for i, c in enumerate(text))
		return text

	def __getitem__(self, idx):
		return {
			self.blank_idx: self.blank, self.repeat_idx: self.repeat, self.space_idx: self.space
		}.get(idx) or (
			self.alphabet[idx] if self.bpe is None else
			self.bpe.IdToPiece(idx).replace(self.space_sentencepiece,
											self.space).replace(self.unk_sentencepiece, self.unk)
		)

	def __len__(self):
		return len(self.alphabet if self.bpe is None else self.bpe) + len([self.repeat, self.space, self.blank])

	def __str__(self):
		return self.alphabet + ''.join([self.repeat, self.space, self.blank])


class Language:
	def __new__(cls, lang):
		return importlib.import_module(lang)
