import argparse
import Levenshtein

class Alignment(object):
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
		self.mode = Alignment.SCORE_UNIFORM

	def set_score(self, score_null=None, score_sub=None, score_del=None, score_ins=None):
		if score_null is not None:
			self.score_null = score_null
		if score_sub is not None:
			self.score_sub = score_sub
		if score_del is not None:
			self.score_del = score_del
		if score_ins is not None:
			self.score_ins = score_ins

	def match(self, a, b):
		if a == b and self.mode == Alignment.SCORE_UNIFORM:
			return self.score_null
		elif self.mode == Alignment.SCORE_UNIFORM:
			return self.score_sub
		elif a == b:
			return self.score_null * len(a)
		else:
			return self.score_sub * len(a)

	def delete(self, a):
		"""
		deleted elements are on seqa
		"""
		if self.mode == Alignment.SCORE_UNIFORM:
			return self.score_del
		return self.score_del * len(a)

	def insert(self, a):
		"""
		inserted elements are on seqb
		"""
		if self.mode == Alignment.SCORE_UNIFORM:
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


class Needleman(Alignment):
	def __init__(self, *args):
		super(Needleman, self).__init__()
		self.semi_global = False
		self.matrix = None

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
			last_col_max, val = max(enumerate([row[-1] for row in self.matrix]), key=lambda a: a[1])
			last_row_max, val = max(enumerate([col for col in self.matrix[-1]]), key=lambda a: a[1])

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

	def align(self, seq_a, seq_b, semi_global=True, mode=None):
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


class Hirschberg(Alignment):
	def __init__(self):
		super(Hirschberg, self).__init__()
		self.needleman = Needleman()

	def last_row(self, seqa, seqb):
		lena = len(seqa)
		lenb = len(seqb)
		pre_row = [0] * (lenb + 1)
		cur_row = [0] * (lenb + 1)

		for j in range(1, lenb + 1):
			pre_row[j] = pre_row[j - 1] + self.insert(seqb[j - 1])

		for i in range(1, lena + 1):
			cur_row[0] = self.delete(seqa[i - 1]) + pre_row[0]
			for j in range(1, lenb + 1):
				score_sub = pre_row[j - 1] + self.match(seqa[i - 1], seqb[j - 1])
				score_del = pre_row[j] + self.delete(seqa[i - 1])
				score_ins = cur_row[j - 1] + self.insert(seqb[j - 1])
				cur_row[j] = max(score_sub, score_del, score_ins)

			pre_row = cur_row
			cur_row = [0] * (lenb + 1)

		return pre_row

	def align_rec(self, seq_a, seq_b):
		aligned_a, aligned_b = [], []
		len_a, len_b = len(seq_a), len(seq_b)

		if len_a == 0:
			for i in range(len_b):
				aligned_a.append(self.separator * len(seq_b[i]))
				aligned_b.append(seq_b[i])
		elif len_b == 0:
			for i in range(len_a):
				aligned_a.append(seq_a[i])
				aligned_b.append(self.separator * len(seq_a[i]))

		elif len(seq_a) == 1:
			aligned_a, aligned_b = self.needleman.align(seq_a, seq_b)

		else:
			mid_a = int(len_a / 2)

			rowleft = self.last_row(seq_a[:mid_a], seq_b)
			rowright = self.last_row(seq_a[mid_a:][::-1], seq_b[::-1])

			rowright.reverse()

			row = [l + r for l, r in zip(rowleft, rowright)]
			maxidx, maxval = max(enumerate(row), key=lambda a: a[1])

			mid_b = maxidx

			aligned_a_left, aligned_b_left = self.align_rec(seq_a[:mid_a], seq_b[:mid_b])
			aligned_a_right, aligned_b_right = self.align_rec(seq_a[mid_a:], seq_b[mid_b:])
			aligned_a = aligned_a_left + aligned_a_right
			aligned_b = aligned_b_left + aligned_b_right

		return aligned_a, aligned_b

	def align(self, seq_a, seq_b, mode=None):
		self.seq_a = seq_a
		self.seq_b = seq_b
		self.len_a = len(self.seq_a)
		self.len_b = len(self.seq_b)
		if mode is not None:
			self.mode = mode
		return self.align_rec(self.seq_a, self.seq_b)

def analyze(ref, hyp, phonetic_replace_groups = []):
	ref0, hyp0 = ref, hyp
	ref, hyp = Needleman().align(list(ref), list(hyp))
	r, h = '', ''
	i = 0
	while i < len(ref):
		if i + 1 < len(hyp) and ref[i] == '|' and hyp[i + 1] == '|':
			r += ref[i + 1]
			h += hyp[i]
			i += 2
		elif i + 1 < len(ref) and ref[i + 1] == '|' and hyp[i] == '|':
			r += ref[i]
			h += hyp[i + 1]
			i += 2
		else:
			r += ref[i]
			h += hyp[i]
			i += 1

	def words():
		k = None
		for i in range(1 + len(r)):
			if i == len(r) or r[i] == ' ':
				yield r[k : i], h[k : i]
				k = None
			elif r[i] != '|' and r[i] != ' ' and k is None:
				k = i

	assert len(r) == len(h)
	phonetic_group = lambda c: ([i for i, g in enumerate(phonetic_replace_groups) if c in g] + [c])[0]
	a = dict(
		chars = dict(
			ok = sum(1 if r[i] == h[i] else 0 for i in range(len(r))), 
			replace = sum(1 if r[i] != '|' and r[i] != h[i] and h[i] != '|' else 0 for i in range(len(r))),
			replace_phonetic = sum(1 if r[i] != '|' and r[i] != h[i] and h[i] != '|' and phonetic_group(r[i]) == phonetic_group(h[i]) else 0 for i in range(len(r))), 
			delete = sum(1 if r[i] != '|' and r[i] != h[i] and h[i] == '|' else 0 for i in range(len(r))),
			insert = sum(1 if r[i] == '|' and h[i] != '|' else 0 for i in range(len(r))),
			total = len(r)
		),
		spaces = dict(
			delete = sum(1 if r[i] == ' ' and h[i] != ' ' else 0 for i in range(len(r))),
			insert = sum(1 if h[i] == ' ' and r[i] != ' ' else 0 for i in range(len(r))),
			total = sum(1 if r[i] == ' ' else 0 for i in range(len(r)))
		),
		words = dict(
			missing_prefix = sum(1 if h_[0] in ' |' else 0 for r_, h_ in words()),
			missing_suffix = sum(1 if h_[-1] in ' |' else 0 for r_, h_ in words()),
			ok_prefix_suffix = sum(1 if h_[0] not in ' |' and h_[-1] not in ' |' else 0 for r_, h_ in words()),
			delete = sum(1 if h_.count('|') > len(r_) // 2 else 0 for r_, h_ in words()),
			replace = sum(1 if sum(1 if h_[i] not in ' |' and h_[i] != r_[i] else 0 for i in range(len(r_))) > len(r_) // 2 else 0 for r_, h_ in words()),
			total = sum(1 if c == ' ' else 0 for c in ref) + 1
		),
		alignment = dict(ref = r, hyp = h),
		input = dict(ref = ref0, hyp = hyp0),
		cer = Levenshtein.distance(ref0.replace(' ', ''), hyp0.replace(' ', '')) / len(ref0.replace(' ', ''))

	)
	return a

RU_PHONETIC_REPLACE_GROUPS = ['АО', 'БП', 'ЗСЦ', 'ВФ', 'ГК', 'ДТ', 'ЧЖШЩ', 'ЭЕИ', 'РЛ', 'ЮУ']

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--ref')
	parser.add_argument('--hyp')
	args = parser.parse_args()

	print(analyze(args.ref, args.hyp, phonetic_replace_groups = RU_PHONETIC_REPLACE_GROUPS))
