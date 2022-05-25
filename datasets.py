import os
import typing
import math
import time
import itertools
import text_processing
import importlib
import torch.utils.data
import sentencepiece
import audio
import utils
import transcripts
import shaping
import operator
import typing
import torch.nn.functional as F

def worker_init_fn(worker_id, num_threads = 1):
	utils.set_random_seed(worker_id)
	utils.reset_cpu_threads(num_threads)


class AudioTextDataset(torch.utils.data.Dataset):
	'''
	Arguments
	? speaker_names = ['', 'speaker1', 'speaker2']
	data_paths:
	{"audio_path" : "/path/to/audio.ext"}
	? {"ref" : "ref text"}
	? {"begin" : 1.0, "end" : 3.0}
	? {"channel" : 0}
	? {"speaker" : 1 | "speaker1"}
	Returned from __getitem__:
	{"audio_path" : "/path/to/audi.ext", "ref" : "ref text", "example_id" : "example_id"}
	Returned from get_meta:
	{"audio_path" : "/path/to/audio.ext", "example_id" : "example_id", "begin" : 0.0 | time_missing, "end" : 0.0 | time_misisng, "channel" : 0 | 1 | channel_missing, "speaker" : 1 | 15 | speaker_missing, "meta" : original_example, "ref" : 'ref or empty before normalization'}
	Comments:
	If speaker_names are not set and speakers are not set, uses channel indices as speakers

	channels note:
		for dataset files channel field will be filled from file or missing
		for audio files channel field will be in range of max_num_channels or missing if mono is true
	'''

	DEFAULT_MODE = 'default'
	BATCHED_CHANNELS_MODE = 'batched_channels'
	BATCHED_TRANSCRIPT_MODE = 'batched_transcript'

	def __init__(
			self,
			data_paths: typing.List[str],
			text_pipelines: typing.List[text_processing.ProcessingPipeline],
			sample_rate: int,
			mode: str = DEFAULT_MODE,
			frontend: typing.Optional[torch.nn.Module] = None,
			speaker_names: typing.Optional[typing.List[str]] = None,
			max_audio_file_size: typing.Optional[float] = None, #bytes
			min_duration: typing.Optional[float] = None,
			max_duration: typing.Optional[float] = None,
			max_num_channels: int = 2,
			mono: bool = True,
			audio_dtype: str = 'float32',
			time_padding_multiple: int = 1,
			audio_backend: typing.Optional[str] = None,
			exclude: typing.Optional[typing.Set] = None,
			bucket_fn: typing.Callable[[typing.List[typing.Dict]], int] = lambda transcript: 0,
			pop_meta: bool = False,
			string_array_encoding: str = 'utf_16_le',
			_print: typing.Callable = print,
			debug_short_long_records_features_from_whole_normalized_signal: bool = False,
			duration_from_transcripts: bool = False,
	):
		self.debug_short_long_records_features_from_whole_normalized_signal = debug_short_long_records_features_from_whole_normalized_signal
		self.mode = mode
		self.min_duration = min_duration
		self.max_duration = max_duration
		self.max_audio_file_size = max_audio_file_size
		self.text_pipelines = text_pipelines
		self.frontend = frontend
		self.sample_rate = sample_rate
		self.time_padding_multiple = time_padding_multiple
		self.mono = mono
		self.audio_backend = audio_backend
		self.audio_dtype = audio_dtype

		data_paths = data_paths if isinstance(data_paths, list) else [data_paths]

		data_paths_ = []
		for data_path in data_paths:
			if os.path.isdir(data_path):
				data_paths_.extend(os.path.join(data_path, filename) for filename in filter(audio.is_audio, os.listdir(data_path)))
			else:
				data_paths_.append(data_path)
		data_paths = data_paths_

		tic = time.time()

		segments = []
		for path in data_paths:
			if audio.is_audio(path):
				assert self.mono or self.mode != AudioTextDataset.DEFAULT_MODE, 'Only mono audio files allowed as dataset input in default mode'
				if self.mono:
					transcript = [dict(audio_path = path, channel = transcripts.channel_missing)]
				else:
					transcript = [dict(audio_path = path, channel = c) for c in range(max_num_channels)]
			else:
				transcript = transcripts.load(path)
			segments.extend(transcript)

		_print('Dataset reading time: ', time.time() - tic)
		tic = time.time()

		# get_or_else required because dictionary could contain None values which we want to replace.
		# dict.get doesnt work in this case
		get_or_else = lambda dictionary, key, default: dictionary[key] if dictionary.get(key) is not None else default
		for t in segments:
			t['ref'] = get_or_else(t, 'ref', transcripts.ref_missing)
			t['begin'] = get_or_else(t, 'begin', transcripts.time_missing)
			t['end'] = get_or_else(t, 'end', transcripts.time_missing)
			t['channel'] = get_or_else(t, 'channel', transcripts.channel_missing) if not self.mono else transcripts.channel_missing

		transcripts.collect_speaker_names(segments, speaker_names = speaker_names or [], num_speakers = max_num_channels, set_speaker_data = True)

		buckets = []
		grouped_segments = []
		transcripts_len = []
		speakers_len = []
		if self.mode == AudioTextDataset.DEFAULT_MODE:
			groupped_transcripts = ((i, [t]) for i, t in enumerate(segments))
		else:
			groupped_transcripts = itertools.groupby(sorted(segments, key = transcripts.group_key), transcripts.group_key)

		for group_key, transcript in groupped_transcripts:
			transcript = sorted(transcript, key = transcripts.sort_key)
			if self.mode == AudioTextDataset.BATCHED_CHANNELS_MODE:
				transcript = transcripts.join_transcript(transcript, self.mono, duration_from_transcripts=duration_from_transcripts)

			if exclude is not None:
				allowed_audio_names = set(transcripts.audio_name(t) for t in transcript if transcripts.audio_name(t) not in exclude)
			else:
				allowed_audio_names = None

			transcript = transcripts.prune(transcript,
			                               allowed_audio_names = allowed_audio_names,
			                               duration = (min_duration if min_duration is not None else 0.0,
			                                           max_duration if max_duration is not None else 24.0 * 3600,), #24h
			                               max_audio_file_size = max_audio_file_size)
			transcript = list(transcript)
			for t in transcript:
				t['example_id'] = AudioTextDataset.get_example_id(t)

			if len(transcript) == 0:
				continue

			bucket = bucket_fn(transcript)
			for t in transcript:
				t['bucket'] = bucket
				speakers_len.append(len(t['speaker']) if (isinstance(t['speaker'], list)) else 1)
			buckets.append(bucket)
			grouped_segments.extend(transcript)
			transcripts_len.append(len(transcript))


		_print('Dataset construction time: ', time.time() - tic)
		tic = time.time()

		self.bucket = torch.tensor(buckets, dtype = torch.short)
		self.audio_path = utils.TensorBackedStringArray([t['audio_path'] for t in grouped_segments], encoding = string_array_encoding)
		self.ref = utils.TensorBackedStringArray([t['ref'] for t in grouped_segments], encoding = string_array_encoding)
		self.begin = torch.tensor([t['begin'] for t in grouped_segments], dtype = torch.float64)
		self.end = torch.tensor([t['end'] for t in grouped_segments], dtype = torch.float64)
		self.channel = torch.tensor([t['channel'] for t in grouped_segments], dtype = torch.int8)
		self.example_id = utils.TensorBackedStringArray([t['example_id'] for t in grouped_segments], encoding = string_array_encoding)
		if self.mode == AudioTextDataset.BATCHED_CHANNELS_MODE:
			self.speaker = torch.tensor([speaker for t in grouped_segments for speaker in t['speaker']], dtype = torch.int64)
		else:
			self.speaker = torch.tensor([t['speaker'] for t in grouped_segments], dtype = torch.int64)
		self.speaker_len = torch.tensor(speakers_len, dtype = torch.int16)
		self.transcript_cumlen = torch.tensor(transcripts_len, dtype = torch.int16).cumsum(dim = 0, dtype = torch.int64)
		if pop_meta:
			self.meta = {}
		else:
			self.meta = {t['example_id']: t for t in grouped_segments}
		_print('Dataset tensors creation time: ', time.time() - tic)

	def state_dict(self) -> dict:
		return {
			'bucket'            : self.bucket,
			'audio_path'        : self.audio_path,
			'ref'               : self.ref,
			'begin'             : self.begin,
			'end'               : self.end,
			'channel'           : self.channel,
			'speaker'           : self.speaker,
			'example_id'        : self.example_id,
			'meta'              : self.meta,
			'speaker_len'    : self.speaker_len,
			'transcript_cumlen' : self.transcript_cumlen
		}

	def load_state_dict(self, state_dict: dict):
		self.bucket = state_dict['bucket']
		self.audio_path = state_dict['audio_path']
		self.ref = state_dict['ref']
		self.begin = state_dict['begin']
		self.end = state_dict['end']
		self.channel = state_dict['channel']
		self.speaker = state_dict['speaker']
		self.example_id = state_dict['example_id']
		self.meta = state_dict['meta']
		self.speaker_len = state_dict['speaker_len']
		self.transcript_cumlen = state_dict['transcript_cumlen']

	def pop_meta(self):
		meta = self.meta
		self.meta = {}
		return meta

	@staticmethod
	def get_example_id(t):
		return '{{ "audio_path" : "{audio_path}", "begin" : {begin:.04f}, "end" : {end:.04f}, "channel" : {channel} }}'.format(
			audio_path = t['audio_path'], begin = t.get('begin', transcripts.time_missing),
			end = t.get('end', transcripts.time_missing), channel = t.get('channel', transcripts.channel_missing))

	def unpack_transcript(self, index: int):
		if index < 0:
			index += len(self.transcript_cumlen)

		transcript = []
		for i in range(int(self.transcript_cumlen[index - 1]) if index > 0 else 0, int(self.transcript_cumlen[index])):
			ref = self.ref[i]
			speaker = self.speaker[i: i + self.speaker_len[i].item()]
			transcript.append(
				dict(
					audio_path = self.audio_path[i],
					ref = ref,
					begin = float(self.begin[i]),
					end = float(self.end[i]),
					channel = int(self.channel[i]),
					speaker = speaker,
					example_id = self.example_id[i]
				)
			)
		return transcript

	def __getitem__(self, index):
		transcript = self.unpack_transcript(index)

		## signal shape here shaping.CT
		signal: shaping.CT; sample_rate: int
		signal, sample_rate = audio.read_audio(transcript[0]['audio_path'], sample_rate = self.sample_rate, mono = self.mono,
		                                       backend = self.audio_backend, duration = self.max_duration,
		                                       dtype = self.audio_dtype)

		transcript = [t for t in transcript if t['channel'] < len(signal)]

		features = []
		# slicing code in time and channel dimension
		for t in transcript:
			channel = t.pop('channel')
			time_slice = slice(int(t['begin'] * sample_rate) if t['begin'] != transcripts.time_missing else 0,
			                   1 + int(t['end'] * sample_rate) if t['end'] != transcripts.time_missing else signal.shape[1])
			# signal shaping.CT -> segment shaping.1T
			if self.mode == AudioTextDataset.DEFAULT_MODE:
				segment: shaping._T = signal[None, channel, :]  # begin, end meta could be corrupted, thats why we dont use it here
			else:
				segment = signal[None, channel, time_slice]

			if self.frontend is not None:
				# debug_short_long_records_features_from_whole_normalized_signal means apply frontend to whole signal instead of segment
				if self.debug_short_long_records_features_from_whole_normalized_signal:
					segment_features = self.frontend(segment)
					hop_length = self.frontend.hop_length
					segment_features = segment_features[:, :, time_slice.start // hop_length:time_slice.stop // hop_length]
					features.append(segment_features.squeeze(0))
				else:
					# max_len = int(math.ceil(segment.shape[1] / 128)) * 128
					# padded_segment = torch.zeros(1, max_len)
					# padded_segment[0, :segment.shape[1]] = segment[0, :]
					features.append(self.frontend(segment, path=transcript[0]['audio_path']).squeeze(0))

					# print(f'PADDED segment from {segment.shape} to {padded_segment.shape}')
			else:
				features.append(segment)

		# speaker alignment and ref processing code
		targets = []
		speakers = []
		for pipeline in self.text_pipelines:
			encoded_refs, aligned_speakers = AudioTextDataset.encode_transcript(transcript, pipeline)
			targets.append(encoded_refs)
			speakers.append(aligned_speakers)

		# replace speaker separators from ref
		for t in transcript:
			t['ref'] = t['ref'].replace(transcripts.speaker_phrase_separator, ' ')

		# speaker generated for all text pipelines, but for backward compatibility, only first pipeline speakers will be used
		speaker = speakers[0]
		# not batch mode
		if self.mode == AudioTextDataset.DEFAULT_MODE:
			transcript, speaker, features = transcript[0], speaker[0], features[0]
			targets = [target[0] for target in targets]

		return [transcript, speaker, features] + targets

	def __len__(self):
		return len(self.transcript_cumlen)

	def collate_fn(self, batch) -> typing.Tuple[
		typing.List[dict], shaping.BS, shaping.BCT, shaping.B, shaping.BLY, shaping.B]:
		if self.mode != AudioTextDataset.DEFAULT_MODE:
			batch = list(zip(*batch))
		#TODO replace meta_s to example_ids
		meta_s, sample_s, sample_x, *sample_y = batch[0]
		time_padding_multiple = [1, 1, self.time_padding_multiple] + [self.time_padding_multiple] * len(sample_y)
		smax_len, xmax_len, *ymax_len = [
			int(math.ceil(max(b[k].shape[-1] for b in batch) / time_padding_multiple[k])) * time_padding_multiple[k]
			for k in range(1, len(batch[0]))
		]

		meta: typing.List[dict] = [b[0] for b in batch]
		x: shaping.BCT = torch.zeros(len(batch), len(sample_x), xmax_len, dtype = sample_x.dtype)
		y: shaping.BLY = torch.zeros(len(batch), len(sample_y), max(ymax_len), dtype = torch.int64)
		s: shaping.BS = torch.full((len(batch), smax_len), transcripts.speaker_missing, dtype = torch.int64)
		xlen: shaping.B = torch.zeros(len(batch), dtype = torch.float32)
		ylen: shaping.B = torch.zeros(len(batch), len(sample_y), dtype = torch.int64)

		# print(f'PADDED frontend from {[sample_x.shape for meta_s, sample_s, sample_x, *sample_y in batch]} '
		# 	  f'to {x.shape}')

		for k, (meta_s, sample_s, sample_x, *sample_y) in enumerate(batch):
			xlen[k] = sample_x.shape[-1] / x.shape[-1] if x.shape[-1] > 0 else 1.0
			x[k, ..., :sample_x.shape[-1]] = sample_x
			s[k, :sample_s.shape[-1]] = sample_s
			for j, t in enumerate(sample_y):
				y[k, j, :t.shape[-1]] = t
				ylen[k, j] = len(t)

		return (meta, s, x, xlen, y, ylen)

	@staticmethod
	def encode_transcript(transcript, pipeline):
		'''
		Method encode transcript refs and provide speaker vectors aligned with this refs
		'''
		encoded_refs = []
		aligned_speakers = []
		for t in transcript:
			tokens = []
			speaker_labels = []
			ref_by_speaker = t['ref'].split(transcripts.speaker_phrase_separator)
			ref_by_speaker = [ref_by_speaker[0]] + [' ' + ref for ref in ref_by_speaker[1:]]
			assert len(ref_by_speaker) == len(t['speaker']), print(f'Speaker phrases: {ref_by_speaker}, speakers: {t["speaker"]}')
			for speaker_ref, speaker_label in zip(ref_by_speaker, t['speaker']):
				processed = pipeline.preprocess(speaker_ref)
				speaker_tokens = torch.tensor(pipeline.encode([processed])[0], dtype = torch.int64)
				tokens.append(speaker_tokens)
				speaker_labels.append(
					torch.full((len(speaker_tokens),), fill_value = speaker_label, dtype = torch.int64))
			encoded_refs.append(torch.cat(tokens))
			aligned_speakers.append(torch.cat(speaker_labels))
		return encoded_refs, aligned_speakers


class BucketingBatchSampler(torch.utils.data.Sampler):
	def __init__(self, dataset, batch_size = 1, world_size = 1):
		super().__init__(dataset)
		self.world_size = world_size  # this value ensure that `world_size` consecutive batches will be constructed from same bucket
		self.dataset = dataset
		self.batch_size = batch_size
		self.buckets = {k: (self.dataset.bucket == k).nonzero(as_tuple = True)[0]
						for k in self.dataset.bucket.unique()}
		self.batch_idx = 0
		self.set_epoch(epoch = 0)

	def __iter__(self):
		return iter(self.shuffled[self.batch_idx:])

	def __len__(self):
		return len(self.shuffled)

	def set_epoch(self, epoch):
		rng = torch.Generator()
		rng.manual_seed(epoch)

		def shuffle_and_split(g, batch_size):
			required_samples_count = math.ceil(len(g) / (batch_size * self.world_size)) * (
						batch_size * self.world_size)
			extension_indices = torch.randint(0, len(g), size = (required_samples_count - len(g),), generator = rng,
			                                  device = g.device)
			g_extended = torch.cat([g, g[extension_indices]])
			return g_extended[torch.randperm(len(g_extended), generator = rng)].reshape(-1, batch_size)

		batches = torch.cat([shuffle_and_split(g, self.batch_size) for g in self.buckets.values()])
		assert len(batches) % self.world_size == 0
		batch_indices = torch.arange(0, len(batches))
		shuffled_indices = torch.randperm(int(len(batches) / self.world_size), generator = rng)
		if self.world_size > 1:
			group_indices = batch_indices.view(-1, self.world_size)
			shuffled_group_indices = group_indices[shuffled_indices]
			shuffled_batch_indices = shuffled_group_indices.flatten()
		else:
			shuffled_batch_indices = batch_indices[shuffled_indices]
		self.shuffled = batches[shuffled_batch_indices]

		right_batches = torch.load(f'/work/pipelines_over_speech_brain/batches_{epoch}.pt')

		assert torch.equal(self.shuffled, right_batches), f'sampling problem {self.shuffled[0][0], right_batches[0][0]}'
		print(f'BATCH PASSED - {self.shuffled[0][0]}')

	def state_dict(self):
		return dict(batch_idx = self.batch_idx)

	def load_state_dict(self, state_dict):
		self.batch_idx = state_dict['batch_idx']


# https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
class DatasetFromSampler(torch.utils.data.Dataset):
	"""Dataset of indexes from `Sampler`."""

	def __init__(self, sampler: torch.utils.data.Sampler):
		self.sampler = sampler
		self.sampler_list = None

	def __getitem__(self, index: int):
		"""Gets element of the dataset.
		Args:
			index (int): index of the element in the dataset
		Returns:
			Single element by index
		"""
		if self.sampler_list is None:
			self.sampler_list = list(self.sampler)
		return self.sampler_list[index]

	def __len__(self) -> int:
		"""
		Returns:
			int: length of the dataset
		"""
		return len(self.sampler)


class DistributedSamplerWrapper(torch.utils.data.DistributedSampler):
	"""
	Wrapper over `Sampler` for distributed training.
	Allows you to use any sampler in distributed mode.
	It is especially useful in conjunction with
	`torch.nn.parallel.DistributedDataParallel`. In such case, each
	process can pass a DistributedSamplerWrapper instance as a DataLoader
	sampler, and load a subset of subsampled data of the original dataset
	that is exclusive to it.
	.. note::
		Sampler is assumed to be of constant size.
	"""

	def __init__(
			self,
			sampler,
			num_replicas: typing.Optional[int] = None,
			rank: typing.Optional[int] = None,
			shuffle: bool = False,
	):
		"""
		Args:
			sampler: Sampler used for subsampling
			num_replicas (int, optional): Number of processes participating in
			  distributed training
			rank (int, optional): Rank of the current process
			  within ``num_replicas``
			shuffle (bool, optional): If true sampler will shuffle the indices
		"""
		super().__init__(
			DatasetFromSampler(sampler),
			num_replicas = num_replicas,
			rank = rank,
			shuffle = shuffle,
		)
		self.sampler = sampler

	def __iter__(self):
		# comments are specific for BucketingBatchSampler as self.sampler, variable names are kept from Catalyst
		self.dataset = DatasetFromSampler(self.sampler)  # hack for DistributedSampler compatibility
		indexes_of_indexes = super().__iter__()  # type: List[int] # batch indices of BucketingBatchSampler
		subsampler_indexes = self.dataset  # type: List[List[int]] # original example indices
		ddp_sampling_operator = operator.itemgetter(
			*indexes_of_indexes)  # operator to extract rank specific batches from original sampled
		return iter(ddp_sampling_operator(subsampler_indexes))  # type: Iterable[List[int]]

	def state_dict(self):
		return self.sampler.state_dict()

	def load_state_dict(self, state_dict):
		self.sampler.load_state_dict(state_dict)

	def set_epoch(self, epoch):
		super().set_epoch(epoch)
		self.sampler.set_epoch(epoch)

	@property
	def batch_idx(self):
		return self.sampler.batch_idx

	@batch_idx.setter
	def batch_idx(self, value):
		self.sampler.batch_idx = value


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

	# TODO: use https://www.oreilly.com/library/view/python-cookbook/0596001673/ch03s15.html
	# _w_ = lambda w: self.space + word + self.space
	# def replace_stem(acc, u, v):
	#	stem, inflection = self.lang.stem(acc, inflection = True)
	#	return stem.replace(self.space + u, v) + invlection
	#
	# word = _w_(word)
	# word = functools.reduce(lambda acc, uv: acc.replace(*uv), [(_w_(u), _w_(v)) for u, v in self.replace_full_forms.items()], word)
	# word = functools.reduce(lambda acc, uv: acc.replace(*uv), self.replace_subwords_forms.items(), word)
	# word = functools.reduce(lambda acc, uv: acc.replace(*uv), [(_w_(u), self.unk) for u, v in self.replace_full_forms_by_unk], word)
	# word = functools.reduce(lambda acc, uv: replace(acc, *uv), self.replace_stems.items(), word)
	# word = word.translate({c : None for c in self.remove_chars})
	# return word.strip()

	def normalize_text(self, text):
		return self.candidate_sep.join(
			self.space.join(map(self.normalize_word, self.lang_normalize_text(candidate).split(self.space))) for
			candidate in self.split_candidates(text)
		)  # or self.unk

	def encode(self, text, normalize = True):
		normalized = self.normalize_text(text) if normalize else text
		chars = self.split_candidates(normalized)[0]
		return normalized, torch.LongTensor(
			[self.chr2idx[c] if i == 0 or c != chars[i - 1] else self.repeat_idx for i, c in
			 enumerate(chars)] if self.bpe is None else self.bpe.EncodeAsIds(chars))

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
			strip = True,
			key = 'hyp'
	):
		decode_ = lambda i, j: self.postprocess_transcript(
			''.join(self[idx[k]] for k in range(i, j + 1) if replace_repeat is False or k == 0 or idx[k] != idx[k - 1]),
			replace_blank = replace_blank, replace_space = replace_space, replace_repeat = replace_repeat,
			strip = strip)
		speaker_ = lambda i, j: (int(speaker[i:1 + j].max()) if torch.is_tensor(
			speaker) else speaker) if speaker is not None and speakers is None else speakers[
			int(speaker[i:1 + j].max())] if speaker is not None and speakers is not None else None
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
		if phonetic_replace_groups:
			text = text.translate({ord(c): g[0] for g in phonetic_replace_groups for c in g})
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
