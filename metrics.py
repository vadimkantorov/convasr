import os
import math
import collections
import json
import enum
import functools
import torch
import Levenshtein

placeholder = '|'
space = ' '
silence = placeholder + space


def exp_moving_average(avg, val, max = 0, K = 50):
	return (1. / K) * min(val, max) + (1 - 1. / K) * avg


class ErrorTagger(enum.Enum):
	typo_easy = 'typo_easy'
	typo_hard = 'typo_hard'
	missing = 'missing'
	missing_ref = 'missing_ref'
	ok = 'ok'

	@staticmethod
	def tag(hyp, ref, p = 0.5, L = 3):
		e = sum(ch != cr and not (ch == space and cr == placeholder) for ch, cr in zip(hyp, ref))
		ref_placeholders = ref.count(placeholder)
		ref_chars = len(ref) - ref_placeholders
		is_typo = (0 < ref_chars < L and len(hyp) <= L
					) or (e >= 0 and ((hyp.count(placeholder) < p * len(ref) and ref_placeholders < p * len(ref))))
		if hyp == ref:
			err = ErrorTagger.ok
		elif is_typo:
			easy = ref_chars < L or (
				e <= 1 or (
					e == 2 and all(
						ch == cr or i >= len(ref) - 2 or (ch == space and cr == placeholder)
						for i, (ch, cr) in enumerate(zip(hyp, ref))
					)
				)
			)
			err = ErrorTagger.typo_easy if easy else ErrorTagger.typo_hard
		else:
			err = ErrorTagger.missing_ref if ref_placeholders >= p * len(ref) else ErrorTagger.missing

		return err.value, e


class PerformanceMeter(dict):
	def update(self, kwargs, subtag = None):
		for name, value in kwargs.items():
			avg_name = f'performance/{name}_avg' + (f'/{subtag}' if subtag else '')
			max_name = f'performance/{name}_max' + (f'/{subtag}' if subtag else '')
			self[avg_name] = exp_moving_average(self.get(avg_name, 0), value)
			self[max_name] = max(self.get(max_name, 0), value)

	def update_memory_metrics(self, byte_scaler = 1024**3):
		device_count = torch.cuda.device_count()
		total_allocated = 0
		total_reserved = 0
		for i in range(device_count):
			device_stats = torch.cuda.memory_stats(i)
			allocated = device_stats['allocated_bytes.all.peak'] / byte_scaler
			total_allocated += allocated

			reserved = device_stats[f'reserved_bytes.all.peak'] / byte_scaler
			total_reserved += reserved
			self.update(dict(allocated = allocated, reserved = reserved), f'cuda:{i}')

		self.update(dict(allocated = total_allocated, reserved = total_reserved), 'total')

	def update_time_metrics(self, time_ms_data, time_ms_fwd, time_ms_bwd, time_ms_model):
		self.update(
			dict(
				time_data = time_ms_data,
				time_forward = time_ms_fwd,
				time_backward = time_ms_bwd,
				time_iteration = time_ms_data + time_ms_model
			)
		)


class WordTagger(collections.defaultdict):
	vocab_hit = 'vocab_hit'
	vocab_miss = 'vocab_miss'

	def __init__(self, lang, word_tags = {}, vocab = set()):
		self.lang = lang
		self.vocab = vocab
		self.stem2tag = {lang.stem(word): tag for tag, words in word_tags.items() for word in words}

	def __missing__(self, word):
		self[word] = self.get(word) or self.stem2tag.get(self.lang.stem(word))
		return self[word]

	def tag(self, word):
		return [self.vocab_hit if word in self.vocab else self.vocab_miss, self[word]]


def align(hyp, ref, score_sub = -2, score_del = -4, score_ins = -3):
	aligner = Needleman()
	aligner.separator = placeholder
	aligner.score_sub = score_sub
	aligner.score_del = score_del
	aligner.score_ins = score_ins
	ref, hyp = aligner.align(list(ref), list(hyp))
	return ''.join(hyp), ''.join(ref)


def split_by_space(hyp, ref):
	words = []
	k = None
	for i in range(1 + len(ref)):
		if i == len(ref) or ref[i] == space:
			ref_word, hyp_word = ref[k:i], hyp[k:i]
			if ref_word:
				words.append((hyp_word, ref_word))
			k = i + 1
	return words


def align_words(hyp, ref, break_ref = False):
	hyp, ref = map(list, align(hyp, ref))

	words = split_by_space(hyp, ref)

	if break_ref:
		words_ = []
		for hyp_word, ref_word in words:
			ref_charinds = [i for i, c in enumerate(ref_word) if c != placeholder]
			ref_word, hyp_word = list(ref_word), list(hyp_word)
			for i in range(len(ref_word)):
				if (not ref_charinds or i < ref_charinds[0]
					or i > ref_charinds[-1]) and hyp_word[i] == space and ref_word[i] == placeholder:
					ref_word[i] = space
			words_.extend(split_by_space(hyp_word, ref_word))

		words = words_

	word_alignment = [
		dict(hyp = ''.join(hyp), ref = ''.join(ref), error_tag = t, hyp_orig = ''.join(hyp).replace(placeholder, ''), ref_orig = ''.join(ref).replace(placeholder, '')) for hyp,
		ref in words for t,
		e in [ErrorTagger.tag(hyp, ref)]
	]
	return ''.join(hyp), ''.join(ref), word_alignment


class ErrorAnalyzer:
	def __init__(self, word_tagger, configs):
		self.word_tagger = word_tagger
		self.configs = configs

	def analyze(self, *, ref, hyp, labels, audio_path = None, full = False, **kwargs):
		hyp, ref = min((cer(h, r), (h, r)) for r in labels.split_candidates(ref) for h in labels.split_candidates(hyp))[1]
		hyp_postproc, ref_postproc = map(labels.postprocess_transcript, [hyp, ref])

		res = dict(
			labels_name = labels.name,
			audio_path = audio_path,
			audio_name = os.path.basename(audio_path),
			ref = ref,
			hyp = hyp,
			**kwargs
		)

		hyp, ref, word_alignment = align_words(hyp, ref, break_ref = True)  #**config['align_words'])
		for w in word_alignment:
			w['ref_tags'] = set(self.word_tagger.tag(w['ref']))
			w['hyp_tags'] = set(self.word_tagger.tag(w['hyp']))
			w['error_tags'] = set(ErrorTagger.tag(w['hyp'], w['ref']))
			w['cer'] = cer(w['hyp_orig'], w['ref_orig'])

		# error_ok_tags

		# total number of words, number of erorrs, inlcuding hitting vocab

		def filter_words(
			word_alignment,
			word_include_tags = [],
			word_exclude_tags = [],
			error_include_tags = [],
			error_exclude_tags = [],
			**kwargs
		):
			word_include_tags, word_exclude_tags, error_include_tags, error_exclude_tags = map(set, [word_include_tags, word_exclude_tags, error_include_tags, error_exclude_tags])
			res = []
			for w in word_alignment:
				if bool(w['ref_tags'] & word_exclude_tags) or bool(w['error_tags'] & error_exclude_tags):
					continue

				if (word_include_tags and not bool(w['ref_tags'] & word_exclude_tags)) or (
					error_include_tags and not bool(w['error_tags'] & error_include_tags)):
					continue

				res.append(w)
			return res

		def compute_metrics(word_alignment, collapse_repeat = False, phonetic_replace_groups = [], **kwargs):
			num_words = len(word_alignment)
			#TODO: take into account other error tags
			num_words_ok = sum(ErrorTagger.ok not in w['error_tags'] for w in word_alignment)
			wer_wordwise = num_words_ok / num_words if num_words != 0 else 0
			cer_wordwise = sum(w['cer'] for w in word_alignment) / num_words if num_words != 0 else 0

			hyp, ref = space.join(w['hyp_orig'] for w in word_alignment), space.join(w['ref_orig'] for w in word_alignment)
			hyp, ref = map(functools.partial(labels.postprocess_transcript, collapse_repeat = collapse_repeat, phonetic_replace_groups = phonetic_replace_groups), [hyp, ref])
			
			wer_postproc = wer(hyp = hyp, ref = ref)
			cer_postproc = cer(hyp = hyp, ref = ref)
			hyp_der, ref_der = [sum(self.word_tagger.vocab_hit in w[k] for w in word_alignment) / num_words if num_words != 0 else 0 for k in ['hyp_tags', 'ref_tags']]

			return dict(wer_wordwise = wer_wordwise, cer_wordwise = cer_wordwise, num_words = num_words, num_words_ok = num_words_ok, wer = wer_postproc, cer = cer_postproc, ref_der = ref_der, hyp_der = hyp_der)
		
		for config_name, config in self.configs.items():
			res[config_name] = compute_metrics(filter_words(word_alignment, **config), **config)
		res.update(res.pop('default'))

		return res


error_types = [e.value for e in ErrorTagger if e != ErrorTagger.ok]


def analyze(
	*,
	ref,
	hyp,
	labels,
	audio_path = '',
	phonetic_replace_groups = [],
	vocab = set(),
	full = False,
	break_ref_alignment = True,
	**kwargs
):
	hyp, ref = min((cer(h, r), (h, r)) for r in labels.split_candidates(ref) for h in labels.split_candidates(hyp))[1]
	hyp_postproc, ref_postproc = map(functools.partial(labels.postprocess_transcript, collapse_repeat = True), [hyp, ref])
	hyp_phonetic, ref_phonetic = map(functools.partial(labels.postprocess_transcript, phonetic_replace_groups = phonetic_replace_groups), [hyp_postproc, ref_postproc])

	a = dict(
		labels_name = labels.name,
		labels = str(labels),
		audio_path = audio_path,
		audio_name = os.path.basename(audio_path),
		hyp_postrpoc = hyp_postproc,
		ref_postproc = ref_postproc,
		ref = ref,
		hyp = hyp,
		cer = cer(hyp_postproc, ref_postproc),
		wer = wer(hyp_postproc, ref_postproc),
		per = cer(hyp_phonetic, ref_phonetic),
		phonetic = dict(ref = ref_phonetic, hyp = hyp_phonetic),
		der = sum(w in vocab for w in hyp.split()) / (1 + hyp.count(' ')),
		**kwargs
	)

	if full:
		hyp, ref, word_alignment = align_words(hyp, ref, break_ref = break_ref_alignment)
		phonetic_group = lambda c: ([i for i, g in enumerate(phonetic_replace_groups) if c in g] + [c])[0]
		hypref_pseudo = {
			t: (
				' '.join((
					r_ if ErrorTagger.tag(h_, r_)[0] in dict(
						typo_easy = ['typo_easy'],
						typo_hard = ['typo_easy', 'typo_hard'],
						missing = ['missing'],
						missing_ref = ['missing_ref']
					)[t] else h_
				).replace(placeholder, '') for w in word_alignment for r_,
							h_ in [(w['ref'], w['hyp'])]),
				ref.replace(placeholder, '')
			)
			for t in error_types
		}

		errors = {
			t: [dict(hyp = r['hyp'], ref = r['ref']) for r in word_alignment if r['error_tag'] == t]
			for t in error_types
		}

		a.update(
			dict(
				alignment = dict(ref = ref, hyp = hyp),
				words = word_alignment,
				error_stats = dict(
					spaces = dict(
						delete = sum(ref[i] == space and hyp[i] != space for i in range(len(ref))),
						insert = sum(hyp[i] == space and ref[i] != space for i in range(len(ref))),
						total = sum(ref[i] == space for i in range(len(ref)))
					),
					chars = dict(
						ok = sum(ref[i] == hyp[i] for i in range(len(ref))),
						replace = sum(
							ref[i] != placeholder and ref[i] != hyp[i] and hyp[i] != placeholder
							for i in range(len(ref))
						),
						replace_phonetic = sum(
							ref[i] != placeholder and ref[i] != hyp[i] and hyp[i] != placeholder
							and phonetic_group(ref[i]) == phonetic_group(hyp[i]) for i in range(len(ref))
						),
						delete = sum(
							ref[i] != placeholder and ref[i] != hyp[i] and hyp[i] == placeholder
							for i in range(len(ref))
						),
						insert = sum(ref[i] == placeholder and hyp[i] != placeholder for i in range(len(ref))),
						total = len(ref)
					),
					words = dict(
						missing_prefix = sum(w['hyp'][0] in silence for w in word_alignment),
						missing_suffix = sum(w['hyp'][-1] in silence for w in word_alignment),
						ok_prefix_suffix = sum(
							w['hyp'][0] not in silence and w['hyp'][-1] not in silence for w in word_alignment
						),
						delete = sum(w['hyp'].count('|') > len(w['ref']) // 2 for w in word_alignment),
						total = len(word_alignment),
						errors = errors,
					),
				),
				mer = len(errors['missing']) / len(word_alignment),
				cer_easy = cer(*hypref_pseudo['typo_easy']),
				wer_easy = wer(*hypref_pseudo['typo_easy']),
				cer_hard = cer(*hypref_pseudo['typo_hard']),
				cer_missing = cer(*hypref_pseudo['missing'])
			)
		)

	return a


def aggregate(analyzed, p = 0.5):
	stats = dict(
		loss_avg = nanmean(analyzed, 'loss'),
		entropy_avg = nanmean(analyzed, 'entropy'),
		cer_avg = nanmean(analyzed, 'cer'),
		wer_avg = nanmean(analyzed, 'wer'),
		mer_avg = nanmean(analyzed, 'mer'),
		cer_easy_avg = nanmean(analyzed, 'cer_easy'),
		wer_easy_avg = nanmean(analyzed, 'wer_easy'),
		cer_hard_avg = nanmean(analyzed, 'cer_hard'),
		cer_missing_avg = nanmean(analyzed, 'cer_missing'),
		der_avg = nanmean(analyzed, 'der')
	)

	errs = collections.defaultdict(int)
	errs_words = {t: [] for t in error_types}
	for a in analyzed:
		if 'words' in a:
			for hyp, ref in map(lambda b: (b['hyp'], b['ref']), sum(a['error_stats']['words']['errors'].values(), [])):
				t, e = ErrorTagger.tag(hyp, ref)
				e = e if t == 'typo_easy' else -1 if t == 'typo_hard' else -2
				errs[e] += 1
				errs_words[t].append(dict(ref = ref, hyp = hyp))
	stats['errors_distribution'] = dict(collections.OrderedDict(sorted(errs.items())))
	stats.update(errs_words)

	return stats


def nanmean(dictlist, key):
	tensor = torch.FloatTensor([r[key] for r in dictlist if key in r])
	isfinite = torch.isfinite(tensor)
	return float(tensor[isfinite].mean()) if isfinite.any() else -1.0


def quantiles(tensor):
	tensor = tensor.sort().values
	return {k: '{:.2f}'.format(float(tensor[int(len(tensor) * k / 100)])) for k in range(0, 100, 10)}


def cer(hyp, ref, edit_distance = Levenshtein.distance):
	cer_ref_len = len(ref.replace(' ', '')) or 1
	return edit_distance(hyp.replace(' ', '').lower(), ref.replace(' ', '').lower()) / cer_ref_len if hyp != ref else 0


def wer(hyp, ref, edit_distance = Levenshtein.distance):
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


if __name__ == '__main__':
	import argparse
	import datasets
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()

	cmd = subparsers.add_parser('analyze')
	cmd.add_argument('--hyp', required = True)
	cmd.add_argument('--ref', required = True)
	cmd.add_argument('--lang', default = 'ru')
	cmd.set_defaults(
		func = lambda hyp,
		ref,
		lang: print(
			json.dumps(
				analyze(hyp = hyp, ref = ref, labels = datasets.Labels(datasets.Language(lang)), full = True),
				ensure_ascii = False,
				indent = 2,
				sort_keys = True
			)
		)
	)

	cmd = subparsers.add_parser('align')
	cmd.add_argument('--hyp', required = True)
	cmd.add_argument('--ref', required = True)
	cmd.set_defaults(
		func = lambda hyp,
		ref: (
			print('\n'.join(f'{k}: {v}' for k, v in zip(['hyp', 'ref'], align(hyp = hyp, ref = ref)))),
			print('\n'.join(map(str, align_words(hyp, ref, break_ref = True)[-1])))
		)
	)

	args = parser.parse_args()
	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
