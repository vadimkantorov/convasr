import os
import math
import collections
import json
import functools
import typing
import language_processing
import Levenshtein

placeholder = '|'
space = ' '
silence = placeholder + space

replace_placeholder = lambda s, rep = '': s.replace(placeholder, rep)


class ErrorTagger:
	typo_easy = 'typo_easy'
	typo_hard = 'typo_hard'
	missing = 'missing'
	missing_ref = 'missing_ref'
	ok = 'ok'

	error_tags = [typo_easy, typo_hard, missing, missing_ref]

	def tag(self, *, hyp, ref, hyp_tags = [], ref_tags = [], p = 0.5, L = 3, clamp = False):
		errors = sum(ch != cr for ch, cr in zip(hyp, ref) if not (ch == space and cr == placeholder))
		errors_without_placeholder = sum(ch != cr for ch, cr in zip(hyp, ref) if ch not in silence and cr not in silence)
		ok_except_end = all(ch == cr or i >= len(ref) - 2 or (ch == space and cr == placeholder) for i, (ch, cr) in enumerate(zip(hyp, ref)))

		ref_placeholders = ref.count(placeholder)
		ref_chars = len(ref) - ref_placeholders

		hyp_empty = hyp.count(placeholder) == len(hyp)
		ref_empty = ref.count(placeholder) == len(ref)

		hyp_vocab_hit = WordTagger.vocab_hit in hyp_tags or WordTagger.stop in hyp_tags
		ref_stop = WordTagger.stop in ref_tags
		vocab_typo_easy = (ref_empty and hyp_vocab_hit) or (hyp_empty and ref_stop)

		short_typo = len(ref) == 1 or (ref_chars == 0 and len(hyp) < L) or (0 < ref_chars < L and len(hyp) <= L)
		short_few_replacements = ref_chars < L and errors_without_placeholder <= 1

		is_typo = vocab_typo_easy or short_typo or (errors >= 0 and (hyp.count(placeholder) < p * len(ref) and ref_placeholders < p * len(ref)))
		if hyp == ref:
			error_tag = ErrorTagger.ok
		elif is_typo:
			easy = vocab_typo_easy or short_few_replacements or errors <= 1 or (len(ref) > 2 and errors == 2 and ok_except_end) or (len(ref) >= 5 and errors <= 2)
			error_tag = ErrorTagger.typo_easy if easy else ErrorTagger.typo_hard
		else:
			error_tag = ErrorTagger.missing_ref if ref_placeholders >= p * len(ref) else ErrorTagger.missing

		if clamp:
			errors = errors if error_tag == ErrorTagger.typo_easy or error_tag == ErrorTagger.ok else -1 if error_tag == ErrorTagger.typo_hard else -2

		return error_tag, errors


class WordTagger(collections.defaultdict):
	vocab_hit = 'vocab_hit'
	vocab_miss = 'vocab_miss'
	stop = 'stop'

	def __init__(self, stemmer = None, word_tags = {}, vocab = set()):
		self.stemmer = stemmer if stemmer is not None else lambda word: word
		self.vocab = vocab
		self.stem2tag = {self.stemmer(word): tag for tag, words in word_tags.items() for word in words}

	def __missing__(self, word):
		self[word] = self.stem2tag.get(self.stemmer(word))
		return self[word]

	def tag(self, word):
		vocab_tags = [self.vocab_hit if word in self.vocab else self.vocab_miss]
		word_tag = self[word]
		return vocab_tags + ([word_tag] if word_tag else [])

class ErrorAnalyzer:
	def __init__(self, word_tagger = WordTagger(), error_tagger = ErrorTagger(), configs = {}, postprocessors = {}):
		self.word_tagger = word_tagger
		self.error_tagger = error_tagger
		self.configs = configs or dict(default = {})
		self.postprocessors = postprocessors

	def aggregate(self, analyzed, sep = '__', defaults = {}):
		keys_with_number_vals = lambda d: [k for k, v in d.items() if isinstance(v, float) or isinstance(v, int)]

		keys = keys_with_number_vals(analyzed[0])
		for c in self.configs:
			keys.extend([c + sep + k for k in keys_with_number_vals(analyzed[0].get(c, {}))])

		stats = {}
		stats.update(defaults)
		stats.update({k: nanmean(analyzed, k, sep = sep) for k in keys})
		default_config_prefix = 'default' + sep
		default_stats = {}
		for name, value in stats.items():
			if name[:len(default_config_prefix)] == default_config_prefix:
				default_stats[name[len(default_config_prefix):]] = value
		stats.update(default_stats)

		error_chars, error_words = collections.defaultdict(int), []
		for a in analyzed:
			for w in a.get('alignment', []):
				error_tag, errors = self.error_tagger.tag(hyp = w['hyp'], ref = w['ref'], clamp = True)
				error_chars[errors] += 1
				if error_tag != ErrorTagger.ok:
					error_words.append(w)

		stats['errors'] = dict(distribution = dict(collections.OrderedDict(sorted(error_chars.items()))), words = error_words)
		return stats

	def filter_words(
		self,
		word_alignment,
		word_include_tags = [],
		word_exclude_tags = [],
		error_include_tags = [],
		error_exclude_tags = [],
		**kwargs
	):
		word_include_tags, word_exclude_tags, error_include_tags, error_exclude_tags = map(set, [word_include_tags, word_exclude_tags, error_include_tags, error_exclude_tags])
		res = []
		# TODO: maybe remove set calls
		for w in word_alignment:
			if bool(set(w['ref_tags']) & word_exclude_tags) or bool(set(w['error_tags']) & error_exclude_tags):
				continue

			if (word_include_tags and not bool(set(w['ref_tags']) & word_include_tags)) or (
				error_include_tags and not bool(set(w['error_tags']) & error_include_tags)):
				continue

			res.append(w)
		return res

	def compute_wordwise_metrics(self, filtered_alignment : typing.List[dict]) -> dict:
		num_words = len(filtered_alignment)
		num_words_ok = sum(ErrorTagger.ok in w['error_tags'] for w in filtered_alignment)
		num_words_missing = sum(ErrorTagger.missing in w['error_tags'] for w in filtered_alignment)

		return dict(
			num_words = num_words,
			num_words_ok = num_words_ok,
			num_words_missing = num_words_missing,

			mer_wordwise = num_words_missing / num_words if num_words != 0 else 0,
			wer_wordwise = 1.0 - num_words_ok / num_words if num_words != 0 else 0,
			cer_wordwise = sum(w['cer'] for w in filtered_alignment) / num_words if num_words != 0 else 0
		)

	def compute_pseudo_metrics(self, word_alignment : typing.List[dict], filtered_alignment : typing.List[dict], postprocess_transcript : typing.Optional[typing.Callable[..., str]] = None, **kwargs) -> dict:
		'''Corrects FILTERED words, i.e. computes what would metrics be if the FILTERED words are replaced by ground truth'''

		# TODO: use sets?
		hyp_pseudo, ref_pseudo = space.join(w['ref'] if w in filtered_alignment else w['hyp'] for w in word_alignment), space.join(w['ref'] for w in word_alignment)
		hyp_pseudo, ref_pseudo = map(postprocess_transcript, [hyp_pseudo, ref_pseudo])
		cer_pseudo, wer_pseudo = cer(hyp = hyp_pseudo, ref = ref_pseudo), wer(hyp = hyp_pseudo, ref = ref_pseudo)

		return dict(
			cer_pseudo = cer_pseudo,
			wer_pseudo = wer_pseudo
		)

	def compute_filtered_metrics(self, word_alignment : typing.List[dict], filtered_alignment : typing.List[dict], postprocess_transcript : typing.Optional[typing.Callable[..., str]], **kwargs) -> dict:
		'''Corrects NOT FILTERED words, i.e. computes what would metrics be if the NOT FILTERED words are replaced by ground truth'''
		# TODO: use sets?

		hyp_filtered, ref_filtered = space.join(w['hyp'] if w in filtered_alignment else w['ref'] for w in word_alignment), space.join(w['ref'] for w in word_alignment)
		hyp_filtered, ref_filtered = map(postprocess_transcript, [hyp_filtered, ref_filtered])

		return dict(
			cer_filtered = cer(hyp = hyp_filtered, ref = ref_filtered),
			wer_filtered = wer(hyp = hyp_filtered, ref = ref_filtered)
		)

	def compute_vocabness_metrics(self, word_alignment : typing.List[dict], filtered_alignment : typing.List[dict], postprocess_transcript : typing.Optional[typing.Callable[..., str]], **kwargs) -> dict:
		num_words = len(filtered_alignment)
		hyp_vocabness, ref_vocabness = [sum(self.word_tagger.vocab_hit in w[k] for w in filtered_alignment) / num_words if num_words != 0 else 0 for k in ['hyp_tags', 'ref_tags']]
		return dict(
			ref_vocabness = ref_vocabness,
			hyp_vocabness = hyp_vocabness
		)

	def analyze(self, hyp : str, ref : str, text_pipeline: typing.Optional[language_processing.ProcessingPipeline] = None, detailed : bool = False, extra : dict = {}, split_candidates : typing.Optional[typing.Callable[[str], typing.List[str]]] = None) -> dict:
		if split_candidates is None:
			split_candidates = lambda s: [s]

		hyp, ref = min((cer(hyp = h, ref = r), (h, r)) for r in split_candidates(ref) for h in split_candidates(hyp))[1]

		# some default options were already chosen
		#TODO: hyp_postproc, ref_postproc = map(postprocess_transcript, [hyp, ref])

		postproc_ref = text_pipeline.postprocess(ref) if text_pipeline is not None else ref
		postproc_hyp = text_pipeline.postprocess(hyp) if text_pipeline is not None else hyp

		# TODO: document common choices for extra
		res = dict(
			ref=postproc_ref,
			hyp=postproc_hyp,
			ref_orig = ref,
			hyp_orig = hyp,
			cer = cer(hyp = postproc_hyp, ref = postproc_ref),
			wer = wer(hyp = postproc_hyp, ref = postproc_ref),
			**extra
		)

		if detailed:
			_hyp_, _ref_ = align_strings(hyp = postproc_hyp, ref = postproc_ref)
			word_alignment = align_words(_hyp_ = _hyp_, _ref_ = _ref_, word_tagger = self.word_tagger, error_tagger = self.error_tagger, compute_cer = True)
			#TODO: rename into words
			res['alignment'] = word_alignment
			char_stats = dict(
				ok = 0, replace = 0, delete = 0, insert = 0, delete_spaces = 0, insert_spaces = 0, total_spaces = 0)
			for ch, cr in zip(_hyp_, _ref_):
				char_stats['ok'] += (cr == ch)
				char_stats['replace'] += (cr != placeholder and cr != ch and ch != placeholder)
				char_stats['delete'] += (cr != placeholder and cr != ch and ch == placeholder)
				char_stats['insert'] += (cr == placeholder and ch != placeholder)
				char_stats['delete_spaces'] += (cr == space and ch != space)
				char_stats['insert_spaces'] += (ch == space and cr != space)
				char_stats['total_spaces'] += (cr == space)
			res['char_stats'] = char_stats

			for config_name, config in self.configs.items():
				config_postprocessor = self.postprocessors[config['postprocessor']] if 'postprocessor' in config else lambda word: word
				filtered_alignment = self.filter_words(word_alignment, **config)
				res[config_name] = self.compute_wordwise_metrics(filtered_alignment = filtered_alignment)

				for m in [self.compute_filtered_metrics, self.compute_pseudo_metrics, self.compute_vocabness_metrics]:
					res[config_name].update(m(word_alignment, filtered_alignment, config_postprocessor, **config))

		return res


def extract_metric_value(analysis_result: dict, key : str, sep : str = '.', missing: typing.Optional[float] = None) -> typing.Optional[float]:
	keys = key.split(sep)
	assert len(keys) <= 2
	value = analysis_result
	for _key in keys:
		try:
			value = value[_key]
		except KeyError:
			return missing
	return value


def nanmean(list_of_dicts : typing.List[dict], key : str, sep : str = '.', missing: float = -1.0) -> float:
	vals = []
	for analysis_result in list_of_dicts:
		val = extract_metric_value(analysis_result, key, sep)
		if val is not None and math.isfinite(val):
			vals.append(val)
	return sum(vals) / len(vals) if vals else missing


def quantiles(vals):
	vals = list(sorted(vals))
	return {k: '{:.2f}'.format(float(vals[int(len(vals) * k / 100)])) for k in range(0, 100, 10)}


def align_words(_hyp_ : str, _ref_: str, word_tagger : WordTagger = WordTagger(), error_tagger : ErrorTagger = ErrorTagger(), postproc : bool = True, compute_cer : bool = False) -> typing.Tuple[str, str, typing.List[dict]]:
	# _hyp_, _ref_ below stand for a pair of aligned strings
	assert len(_hyp_) == len(_ref_)

	def split_by_space_into_word_pairs(*, _hyp_ : str, _ref_ : str, copy_space = False) -> typing.List[typing.Tuple[str, str]]:
		assert len(_hyp_) == len(_ref_)
		hyp, ref = list(_hyp_), list(_ref_)

		# copy spaces from hyp to ref outside the ref
		ref_charinds = [i for i, c in enumerate(ref) if c != placeholder]
		for i in range(len(ref)):
			if (not ref_charinds or i < ref_charinds[0]
				or i > ref_charinds[-1]) and hyp[i] == space and ref[i] == placeholder:
				ref[i] = space

		if copy_space:

			# replace placeholders by spaces around the ref word
			if ref_charinds:
				before = ref_charinds[0] - 1
				after = ref_charinds[-1] + 1

				hyp_, ref_ = replace_placeholder(''.join(hyp)), replace_placeholder(''.join(ref))
				if hyp_.endswith(ref_) and before >= 0 and hyp[before] not in silence:
					ref[before] = space
				if hyp_.startswith(ref_) and after < len(hyp) and hyp[after] not in silence:
					ref[after] = space

		ref += [space]
		hyp += [space]
		k, words = 0, []

		for i in range(len(ref)):
			ipp = i + 1
			if ref[i] == space:

				l = ipp
				if hyp[i] in silence:
					j = i
				else:
					left = ref_charinds and i < ref_charinds[0]
					if left:
						j = ipp
						l = ipp
					else:
						j = i
						l = i
					ref[i] = placeholder

				if k != j:
					words.append((''.join(hyp[k:j]), ''.join(ref[k:j])))

				k = l

		return words

	def prefer_replacement(*, hyp, ref):
		hyp, ref = list(hyp), list(ref)
		for k in range(len(ref) - 1):
			if ref[k] == placeholder and hyp[k] != placeholder and ref[k + 1] != placeholder and hyp[k + 1] == placeholder:
				ref[k] = ref[k + 1]
				ref[k + 1] = placeholder
			elif hyp[k] == placeholder and ref[k] != placeholder and hyp[k + 1] != placeholder and ref[k + 1] == placeholder:
				hyp[k] = hyp[k + 1]
				hyp[k + 1] = placeholder
		hyp, ref = zip(*[(ch, cr) for ch, cr in zip(hyp, ref) if not (cr == ch == placeholder)])
		return ''.join(hyp), ''.join(ref)

	hyp_ref_word_pairs = split_by_space_into_word_pairs(_hyp_ = _hyp_, _ref_ = _ref_, copy_space = False)

	if postproc:
		tmp_word_pairs = []
		for i, (hyp_word, ref_word) in enumerate(hyp_ref_word_pairs):
			assert len(hyp_word) == len(ref_word)
			hyp_word, ref_word = prefer_replacement(hyp = hyp_word, ref = ref_word)
			tmp_word_pairs.extend(split_by_space_into_word_pairs(_hyp_ = hyp_word, _ref_ = ref_word, copy_space = True))
		hyp_ref_word_pairs = tmp_word_pairs

	word_alignment = []
	for hyp_word, ref_word in hyp_ref_word_pairs:
		assert len(hyp_word) == len(ref_word)
		w = dict(
			_hyp_ = hyp_word,
			_ref_ = ref_word,
			hyp = replace_placeholder(hyp_word),
			ref = replace_placeholder(ref_word)
		)
		w['ref_tags'] = word_tagger.tag(w['ref'])
		w['hyp_tags'] = word_tagger.tag(w['hyp'])
		#TODO: unify .tag() API
		w['error_tags'] = [error_tagger.tag(hyp = w['hyp'], ref = w['ref'], hyp_tags = w['hyp_tags'], ref_tags = w['ref_tags'])[0]]

		#TODO: remove error_tag
		w['error_tag'] = w['error_tags'][0]

		w['len'] = len(w['ref'])
		if compute_cer:
			w['cer'] = cer(hyp = w['hyp'], ref = w['ref'])

		word_alignment.append(w)

	return word_alignment


def align_strings(*, hyp : str, ref : str, score_sub : int = -2, score_del : int = -4, score_ins : int = -3) -> typing.Tuple[str, str]:
	aligner = Needleman()
	aligner.separator = placeholder
	aligner.score_sub = score_sub
	aligner.score_del = score_del
	aligner.score_ins = score_ins
	#TODO: are conversions to list needed?
	ref, hyp = aligner.align(list(ref), list(hyp))
	assert len(ref) == len(hyp)

	return ''.join(hyp), ''.join(ref)

def cer(*, hyp, ref, edit_distance = Levenshtein.distance):
	cer_ref_len = len(ref.replace(' ', '')) or 1
	return edit_distance(hyp.replace(' ', '').lower(), ref.replace(' ', '').lower()) / cer_ref_len if hyp != ref else 0


def wer(*, hyp, ref, edit_distance = Levenshtein.distance):
	# build mapping of words to integers, Levenshtein package only accepts strings
	b = set(hyp.split() + ref.split())
	word2char = dict(zip(b, range(len(b))))
	wer_ref_len = len(ref.split()) or 1
	return edit_distance(
		''.join([chr(word2char[w]) for w in hyp.split()]), ''.join([chr(word2char[w]) for w in ref.split()])
	) / wer_ref_len if hyp != ref else 0


def levenshtein(a, b):
	"""Calculates the Levenshtein distance between a and b.
	The code was copied from: http://hetland.org/coding/python/levenshtein.py
	"""
	n, m = len(a), len(b)
	if n > m:
		# Make sure n <= m, to use O(min(n,m)) space
		a, b = b, a
		n, m = m, n

	current = list(range(n + 1))
	for i in range(1, m + 1):
		previous, current = current, [i] + [0] * n
		for j in range(1, n + 1):
			add, delete = previous[j] + 1, current[j - 1] + 1
			change = previous[j - 1]
			if a[j - 1] != b[i - 1]:
				change = change + 1
			current[j] = min(add, delete, change)

	return current[n]


class Needleman:
	# taken from https://github.com/leebird/alignment/blob/master/alignment/alignment.py
	SCORE_UNIFORM = 1
	SCORE_PROPORTION = 2

	def __init__(self):
		self.seq_a = None
		self.seq_b = None
		self.len_a = None
		self.len_b = None
		self.score_null = 5
		self.score_sub = -100
		self.score_del = -3
		self.score_ins = -3
		self.separator = '|'
		self.mode = self.SCORE_UNIFORM
		self.semi_global = False
		self.matrix = None

	def set_score(self, score_null = None, score_sub = None, score_del = None, score_ins = None):
		if score_null is not None:
			self.score_null = score_null
		if score_sub is not None:
			self.score_sub = score_sub
		if score_del is not None:
			self.score_del = score_del
		if score_ins is not None:
			self.score_ins = score_ins

	def match(self, a, b):
		if a == b and self.mode == self.SCORE_UNIFORM:
			return self.score_null
		elif self.mode == self.SCORE_UNIFORM:
			return self.score_sub
		elif a == b:
			return self.score_null * len(a)
		else:
			return self.score_sub * len(a)

	def delete(self, a):
		"""
		deleted elements are on seqa
		"""
		if self.mode == self.SCORE_UNIFORM:
			return self.score_del
		return self.score_del * len(a)

	def insert(self, a):
		"""
		inserted elements are on seqb
		"""
		if self.mode == self.SCORE_UNIFORM:
			return self.score_ins
		return self.score_ins * len(a)

	def score(self, aligned_seq_a, aligned_seq_b):
		score = 0
		for a, b in zip(aligned_seq_a, aligned_seq_b):
			if a == b:
				score += self.score_null
			else:
				if a == self.separator:
					score += self.score_ins
				elif b == self.separator:
					score += self.score_del
				else:
					score += self.score_sub
		return score

	def map_alignment(self, aligned_seq_a, aligned_seq_b):
		map_b2a = []
		idx = 0
		for x, y in zip(aligned_seq_a, aligned_seq_b):
			if x == y:
				# if two positions are the same
				map_b2a.append(idx)
				idx += 1
			elif x == self.separator:
				# if a character is inserted in b, map b's
				# position to previous index in a
				# b[0]=0, b[1]=1, b[2]=1, b[3]=2
				# aa|bbb
				# aaabbb
				map_b2a.append(idx)
			elif y == self.separator:
				# if a character is deleted in a, increase
				# index in a, skip this position
				# b[0]=0, b[1]=1, b[2]=3
				# aaabbb
				# aa|bbb
				idx += 1
				continue
		return map_b2a

	def init_matrix(self):
		rows = self.len_a + 1
		cols = self.len_b + 1
		self.matrix = [[0] * cols for i in range(rows)]

	def compute_matrix(self):
		seq_a = self.seq_a
		seq_b = self.seq_b
		len_a = self.len_a
		len_b = self.len_b

		if not self.semi_global:
			for i in range(1, len_a + 1):
				self.matrix[i][0] = self.delete(seq_a[i - 1]) + self.matrix[i - 1][0]
			for i in range(1, len_b + 1):
				self.matrix[0][i] = self.insert(seq_b[i - 1]) + self.matrix[0][i - 1]

		for i in range(1, len_a + 1):
			for j in range(1, len_b + 1):
				"""
				Note that rows = len_a+1, cols = len_b+1
				"""

				score_sub = self.matrix[i - 1][j - 1] + self.match(seq_a[i - 1], seq_b[j - 1])
				score_del = self.matrix[i - 1][j] + self.delete(seq_a[i - 1])
				score_ins = self.matrix[i][j - 1] + self.insert(seq_b[j - 1])
				self.matrix[i][j] = max(score_sub, score_del, score_ins)

	def backtrack(self):
		aligned_seq_a, aligned_seq_b = [], []
		seq_a, seq_b = self.seq_a, self.seq_b

		if self.semi_global:
			# semi-global settings, len_a = row numbers, column length, len_b = column number, row length
			last_col_max, val = max(enumerate([row[-1] for row in self.matrix]), key = lambda a: a[1])
			last_row_max, val = max(enumerate([col for col in self.matrix[-1]]), key = lambda a: a[1])

			if self.len_a < self.len_b:
				i, j = self.len_a, last_row_max
				aligned_seq_a = [self.separator] * (self.len_b - last_row_max)
				aligned_seq_b = seq_b[last_row_max:]
			else:
				i, j = last_col_max, self.len_b
				aligned_seq_a = seq_a[last_col_max:]
				aligned_seq_b = [self.separator] * (self.len_a - last_col_max)
		else:
			i, j = self.len_a, self.len_b

		mat = self.matrix

		while i > 0 or j > 0:
			# from end to start, choose insert/delete over match for a tie
			# why?
			if self.semi_global and (i == 0 or j == 0):
				if i == 0 and j > 0:
					aligned_seq_a = [self.separator] * j + aligned_seq_a
					aligned_seq_b = seq_b[:j] + aligned_seq_b
				elif i > 0 and j == 0:
					aligned_seq_a = seq_a[:i] + aligned_seq_a
					aligned_seq_b = [self.separator] * i + aligned_seq_b
				break

			if j > 0 and mat[i][j] == mat[i][j - 1] + self.insert(seq_b[j - 1]):
				aligned_seq_a.insert(0, self.separator * len(seq_b[j - 1]))
				aligned_seq_b.insert(0, seq_b[j - 1])
				j -= 1

			elif i > 0 and mat[i][j] == mat[i - 1][j] + self.delete(seq_a[i - 1]):
				aligned_seq_a.insert(0, seq_a[i - 1])
				aligned_seq_b.insert(0, self.separator * len(seq_a[i - 1]))
				i -= 1

			elif i > 0 and j > 0 and mat[i][j] == mat[i - 1][j - 1] + self.match(seq_a[i - 1], seq_b[j - 1]):
				aligned_seq_a.insert(0, seq_a[i - 1])
				aligned_seq_b.insert(0, seq_b[j - 1])
				i -= 1
				j -= 1

			else:
				print(seq_a)
				print(seq_b)
				print(aligned_seq_a)
				print(aligned_seq_b)
				# print(mat)
				raise Exception('backtrack error', i, j, seq_a[i - 2:i + 1], seq_b[j - 2:j + 1])
				pass

		return aligned_seq_a, aligned_seq_b

	def align(self, seq_a, seq_b, semi_global = True, mode = None):
		self.seq_a = seq_a
		self.seq_b = seq_b
		self.len_a = len(self.seq_a)
		self.len_b = len(self.seq_b)

		self.semi_global = semi_global

		# 0: left-end 0-penalty, 1: right-end 0-penalty, 2: both ends 0-penalty
		# self.semi_end = semi_end

		if mode is not None:
			self.mode = mode
		self.init_matrix()
		self.compute_matrix()
		return self.backtrack()


def cmd_analyze(hyp, ref, val_config, vocab, lang, detailed):
	vocab = set(map(str.strip, open(vocab))) if os.path.exists(vocab) else set()
	if lang is not None:
		import datasets
		import ru

		labels = {
			'ru': lambda: datasets.Labels(ru)
		}
		postprocess_transcript = labels[lang]().postprocess_transcript
	else:
		postprocess_transcript = False
	if os.path.exists(val_config):
		val_config = json.load(open(val_config))
		analyzer_configs = val_config['error_analyzer']
		word_tags = val_config['word_tags']
	else:
		analyzer_configs = {}
		word_tags = {}

	word_tagger = WordTagger(word_tags = word_tags, vocab = vocab)
	error_tagger = ErrorTagger()
	analyzer = ErrorAnalyzer(word_tagger = word_tagger, error_tagger = error_tagger, configs = analyzer_configs)
	report = analyzer.analyze(hyp = hyp, ref = ref, postprocess_transcript = postprocess_transcript, detailed=detailed)
	print(json.dumps(report, ensure_ascii = False, indent = 2, sort_keys = True))


def cmd_align(hyp, ref):
	alignment = align_strings(hyp=hyp, ref=ref)
	print('\n'.join(f'{k}: {v}' for k, v in zip(['hyp', 'ref'], alignment)))
	print('\n'.join(map(str, align_words(alignment))))


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()

	cmd = subparsers.add_parser('analyze')
	cmd.add_argument('--hyp', required = True)
	cmd.add_argument('--ref', required = True)
	cmd.add_argument('--lang')
	cmd.add_argument('--detailed', action='store_true')
	cmd.add_argument('--vocab', default = 'data/vocab_word_list.txt')
	cmd.add_argument('--val-config', default = 'configs/ru_val_config.json')
	cmd.set_defaults(func=cmd_analyze)

	cmd = subparsers.add_parser('align')
	cmd.add_argument('--hyp', required = True)
	cmd.add_argument('--ref', required = True)
	cmd.set_defaults(func=cmd_align)

	args = parser.parse_args()
	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
