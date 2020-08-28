import os
import math
import collections
import json
import functools
import torch
import Levenshtein
import psutil

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

	def __init__(self, lang = None, word_tags = {}, vocab = set()):
		self.lang = lang
		self.vocab = vocab
		self.stem2tag = {lang.stem(word) if self.lang is not None else word: tag for tag, words in word_tags.items() for word in words}

	def __missing__(self, word):
		self[word] = self.get(word) or self.stem2tag.get(self.lang.stem(word) if self.lang is not None else word)
		return self[word]

	def tag(self, word):
		vocab_tags = [self.vocab_hit if word in self.vocab else self.vocab_miss]
		word_tag = self[word] 
		return vocab_tags + ([word_tag] if word_tag else [])

class ErrorAnalyzer:
	def __init__(self, word_tagger = WordTagger(), error_tagger = ErrorTagger(), configs = {}):
		self.word_tagger = word_tagger
		self.error_tagger = error_tagger
		self.configs = configs or dict(default = {})

	def aggregate(self, analyzed):
		keys = [k for k, v in analyzed[0].items() if isinstance(v, float) or isinstance(v, int)]
		stats = {(c + '__' + k).replace('default__', '') : nanmean(analyzed, (c + '.' + k).replace('default.', '')) for c in self.configs for k in keys}
		error_chars = collections.defaultdict(int)
		error_words = []
		for a in analyzed:
			for w in a.get('alignment', []):
				error_tag, errors = self.error_tagger.tag(hyp = w['hyp'], ref = w['ref'], clamp = True)
				error_chars[errors] += 1
				if error_tag != ErrorTagger.ok:
					error_words.append(w)

		stats['errors'] = dict(distribution = dict(collections.OrderedDict(sorted(error_chars.items()))), words = error_words)
		return stats

	def analyze(self, hyp, ref, full = False, extra = {}, postprocess_transcript = (lambda s, *args, **kwargs: s), split_candidates = (lambda s: [s])):
		# TODO: add error_ok_tags
		# TODO: respect full flag

		hyp, ref = min((cer(hyp = h, ref = r), (h, r)) for r in split_candidates(ref) for h in split_candidates(hyp))[1] 
		hyp_postproc, ref_postproc = map(postprocess_transcript, [hyp, ref])
		res = dict(
			ref = ref,
			hyp = hyp,
			**extra
		)
		_hyp_, _ref_, word_alignment = align_words(hyp = hyp, ref = ref, word_tagger = self.word_tagger, error_tagger = self.error_tagger, compute_cer = True) # **config['align_words']) 

		res['alignment'] = word_alignment
		res['char_stats'] = char_stats = dict(
			ok = 0, replace = 0, delete = 0, insert = 0, delete_spaces = 0, insert_spaces = 0, total_spaces = 0)
		for ch, cr in zip(_hyp_, _ref_):
			char_stats['ok'] += (cr == ch)
			char_stats['replace'] += (cr != placeholder and cr != ch and ch != placeholder)
			char_stats['delete'] += (cr != placeholder and cr != ch and ch == placeholder)
			char_stats['insert'] += (cr == placeholder and ch != placeholder)
			char_stats['delete_spaces'] += (cr == space and ch != space)
			char_stats['insert_spaces'] += (ch == space and cr != space)
			char_stats['total_spaces'] += (cr == space)

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
				if bool(set(w['ref_tags']) & word_exclude_tags) or bool(set(w['error_tags']) & error_exclude_tags):
					continue

				if (word_include_tags and not bool(set(w['ref_tags']) & word_include_tags)) or (
					error_include_tags and not bool(set(w['error_tags']) & error_include_tags)):
					continue

				res.append(w)
			return res

		def compute_metrics(word_alignment, filtered_alignment, collapse_repeat = False, phonetic_replace_groups = [], **kwargs):
			postprocess_transcript_reified = functools.partial(postprocess_transcript, collapse_repeat = collapse_repeat, phonetic_replace_groups = phonetic_replace_groups)
			
			num_words = len(filtered_alignment)
			num_words_ok = sum(ErrorTagger.ok in w['error_tags'] for w in filtered_alignment)
			num_words_missing = sum(ErrorTagger.missing in w['error_tags'] for w in filtered_alignment)
			
			mer_wordwise = num_words_missing / num_words if num_words != 0 else 0
			wer_wordwise = num_words_ok / num_words if num_words != 0 else 0
			cer_wordwise = sum(w['cer'] for w in filtered_alignment) / num_words if num_words != 0 else 0

			hyp_pseudo, ref_pseudo = space.join(w['ref_orig'] if w in filtered_alignment else w['hyp_orig'] for w in word_alignment), space.join(w['ref_orig'] for w in word_alignment)
			hyp_pseudo, ref_pseudo = map(postprocess_transcript_reified, [hyp_pseudo, ref_pseudo])
			cer_pseudo, wer_pseudo = cer(hyp = hyp_pseudo, ref = ref_pseudo), wer(hyp = hyp_pseudo, ref = ref_pseudo)

			hyp_only, ref_only = space.join(w['hyp_orig'] for w in filtered_alignment), space.join(w['ref_orig'] for w in filtered_alignment)
			hyp_only, ref_only = map(postprocess_transcript_reified, [hyp_only, ref_only])
			cer_only, wer_only = cer(hyp = hyp_only, ref = ref_only), wer(hyp = hyp_only, ref = ref_only)
			hyp_der, ref_der = [sum(self.word_tagger.vocab_hit in w[k] for w in filtered_alignment) / num_words if num_words != 0 else 0 for k in ['hyp_tags', 'ref_tags']]

			return dict(cer_wordwise = cer_wordwise, wer_wordwise = wer_wordwise, mer_wordwise = mer_wordwise, num_words = num_words, num_words_ok = num_words_ok, num_words_missing = num_words_missing, ref_der = ref_der, hyp_der = hyp_der, cer = cer_only, wer = wer_only, cer_pseudo = cer_pseudo, wer_pseudo = wer_pseudo)
		
		for config_name, config in self.configs.items():
			filtered_alignment = filter_words(word_alignment, **config)
			res[config_name] = compute_metrics(word_alignment, filtered_alignment, **config)
		
		res.update(res.pop('default'))

		return res


class PerformanceMeter(dict):
	def update(self, kwargs, subtag = None):
		for name, value in kwargs.items():
			avg_name = f'performance/{name}_avg' + (f'/{subtag}' if subtag else '')
			max_name = f'performance/{name}_max' + (f'/{subtag}' if subtag else '')
			self[avg_name] = exp_moving_average(self.get(avg_name, 0), value)
			self[max_name] = max(self.get(max_name, 0), value)

	def update_memory_metrics(self, byte_scaler = 1024**3, measure_pss_ram = False):
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

		if measure_pss_ram:
			process = psutil.Process()
			children = process.children(recursive=True)
			total_pss_ram = process.memory_full_info().pss + sum(
				child.memory_full_info().pss for child in children
			)
			self.update(dict(pss_ram = total_pss_ram / byte_scaler))

	def update_time_metrics(self, time_ms_data, time_ms_fwd, time_ms_bwd, time_ms_model):
		self.update(
			dict(
				time_data = time_ms_data,
				time_forward = time_ms_fwd,
				time_backward = time_ms_bwd,
				time_iteration = time_ms_data + time_ms_model
			)
		)

def nanmean(dictlist, key):
	prefix, suffix = ('', key) if '.' not in key else key.split('.')
	tensor = torch.FloatTensor([r_[suffix] for r in dictlist for r_ in [r.get(prefix, r)] if suffix in r_])
	isfinite = torch.isfinite(tensor)
	return float(tensor[isfinite].mean()) if isfinite.any() else -1.0


def quantiles(tensor):
	tensor = tensor.sort().values
	return {k: '{:.2f}'.format(float(tensor[int(len(tensor) * k / 100)])) for k in range(0, 100, 10)}

def exp_moving_average(avg, val, max = 0, K = 50):
	return (1. / K) * min(val, max) + (1 - 1. / K) * avg


def align_words(*, hyp, ref, word_tagger = WordTagger(), error_tagger = ErrorTagger(), postproc = True, compute_cer = False):
	def split_by_space(*, hyp, ref, copy_space = False):
		assert len(hyp) == len(ref)
		hyp, ref = list(hyp)[:], list(ref)[:]
		
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
					words.append((hyp[k:j], ref[k:j]))
				
				k = l

		return words

	def prefer_replacement(*, hyp, ref):
		hyp, ref = hyp[:], ref[:]
		for k in range(len(ref) - 1):
			if ref[k] == placeholder and hyp[k] != placeholder and ref[k + 1] != placeholder and hyp[k + 1] == placeholder:
				ref[k] = ref[k + 1]
				ref[k + 1] = placeholder
			elif hyp[k] == placeholder and ref[k] != placeholder and hyp[k + 1] != placeholder and ref[k + 1] == placeholder:
				hyp[k] = hyp[k + 1]
				hyp[k + 1] = placeholder
		hyp, ref = zip(*[(ch, cr) for ch, cr in zip(hyp, ref) if not (cr == ch == placeholder)])
		return hyp, ref

	hyp, ref = map(list, align(hyp, ref))
	words = split_by_space(hyp = hyp, ref = ref, copy_space = False)
	if postproc:
		words_ = []
		for i, (hyp_word, ref_word) in enumerate(words):
			hyp_word, ref_word = prefer_replacement(hyp = hyp_word, ref = ref_word)
			words_.extend(split_by_space(hyp = hyp_word, ref = ref_word, copy_space = True))
		words = words_
	
	word_alignment = []
	for hyp, ref in words:
		w = dict(hyp = ''.join(hyp), ref = ''.join(ref), hyp_orig = replace_placeholder(''.join(hyp)), ref_orig = replace_placeholder(''.join(ref)))
		w['ref_tags'] = word_tagger.tag(w['ref']) 
		w['hyp_tags'] = word_tagger.tag(w['hyp']) 
		w['error_tags'] = [error_tagger.tag(hyp = w['hyp'], ref = w['ref'], hyp_tags = w['hyp_tags'], ref_tags = w['ref_tags'])[0]]
		w['error_tag'] = w['error_tags'][0]
		w['len'] = len(w['ref_orig'])
		if compute_cer:
			w['cer'] = cer(hyp = w['hyp_orig'], ref = w['ref_orig'])
		word_alignment.append(w)
	return ''.join(hyp), ''.join(ref), word_alignment


def align(hyp, ref, score_sub = -2, score_del = -4, score_ins = -3):
	aligner = Needleman()
	aligner.separator = placeholder
	aligner.score_sub = score_sub
	aligner.score_del = score_del
	aligner.score_ins = score_ins
	ref, hyp = aligner.align(list(ref), list(hyp))
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


if __name__ == '__main__':
	import argparse
	import datasets
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()

	cmd = subparsers.add_parser('analyze')
	cmd.add_argument('--hyp', required = True)
	cmd.add_argument('--ref', required = True)
	cmd.add_argument('--lang', default = 'ru')
	cmd.add_argument('--vocab', default = 'data/vocab_word_list.txt')
	cmd.add_argument('--val-config', default = 'configs/ru_val_config.json')
	cmd.set_defaults(
		func = lambda hyp,
		ref, val_config, vocab,
		lang: print(
			json.dumps(
				ErrorAnalyzer(**(dict(configs = json.load(open(val_config))['error_analyzer'], word_tagger = WordTagger(word_tags = json.load(open(val_config))['word_tags'], vocab = set(map(str.strip, open(vocab))) if os.path.exists(vocab) else set())) if os.path.exists(val_config) else {})).analyze(hyp = hyp, ref = ref, labels = datasets.Labels(datasets.Language(lang)), full = True),
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
		ref, break_ref: (
			print('\n'.join(f'{k}: {v}' for k, v in zip(['hyp', 'ref'], align(hyp = hyp, ref = ref)))),
			print('\n'.join(map(str, align_words(hyp = hyp, ref = ref)[-1])))
		)
	)

	args = parser.parse_args()
	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
