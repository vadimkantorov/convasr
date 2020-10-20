import importlib

import sentencepiece
import torch

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
		strip = True,
		key = 'hyp'
	):
		decode_ = lambda i, j: self.postprocess_transcript(''.join(self[idx[k]] for k in range(i, j + 1) if replace_repeat is False or k == 0 or idx[k] != idx[k - 1]), replace_blank = replace_blank, replace_space = replace_space, replace_repeat = replace_repeat, strip = strip)
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
		if phonetic_replace_groups:
			text = text.translate({ord(c) : g[0] for g in phonetic_replace_groups for c in g})
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