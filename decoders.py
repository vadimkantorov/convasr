import Levenshtein 
import torch

def compute_cer(s1, s2):
	return Levenshtein.distance(s1.replace(' ', ''), s2.replace(' ', ''))

def compute_wer(s1, s2):
	# build mapping of words to integers, Levenshtein package only accepts strings
	b = set(s1.split() + s2.split())
	word2char = dict(zip(b, range(len(b))))
	return Levenshtein.distance(''.join([chr(word2char[w]) for w in s1.split()]), ''.join([chr(word2char[w]) for w in s2.split()]))

class GreedyDecoder(object):
	def __init__(self, labels):
	   self.labels = labels 

	def decode(self, log_probs, output_lengths):
		decoded_idx = log_probs.argmax(dim = 1).tolist()
		return [[i for k, i in enumerate(d) if (k == 0 or i != d[k - 1]) and i != self.labels.blank_idx] for d in decoded_idx]

class BeamSearchDecoder(object):
	def __init__(self, labels, lm_path, beam_width, beam_alpha = 0, beam_beta = 0, cutoff_top_n = 40, cutoff_prob = 1.0, num_workers = 1):
		import ctcdecode
		self.labels = labels
		self.beam_search_decoder = ctcdecode.CTCBeamDecoder(str(labels).lower(), lm_path, beam_alpha, beam_beta, cutoff_top_n, cutoff_prob, beam_width, num_workers, labels.blank_idx, log_probs_input = True)

	def decode(self, log_probs, output_lengths):
		decoded_chr, decoded_scores, decoded_offsets, decoded_lengths = self.beam_search_decoder.decode(log_probs.permute(0, 2, 1).cpu(), torch.IntTensor(output_lengths))
		decoded_top = decoded_scores.argmax(dim = 1)
		return [d[t][:l[t]].tolist() for d, l, t in zip(decoded_chr, decoded_lengths, decoded_top)]

def unused_levenshtein(a, b):
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

