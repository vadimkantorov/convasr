#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors

import Levenshtein 
import torch

def compute_cer(s1, s2):
	return Levenshtein.distance(s1.replace(' ', ''), s2.replace(' ', ''))

def compute_wer(s1, s2):
	# build mapping of words to integers, Levenshtein package only accepts strings
	word2char = dict(zip(set(s1.split() + s2.split()), range(len(b))))
	return Levenshtein.distance(''.join([chr(word2char[w]) for w in s1.split()]), ''.join([chr(word2char[w]) for w in s2.split()]))

class GreedyDecoder(object):
	def __init__(self, labels):
	   self.labels = labels 

	def decode(self, probs, sizes):
		decoded = probs.argmax(dim = 1).tolist()
		return [[i for k, i in enumerate(d) if (k == 0 or i != d[k - 1]) and i != self.labels.blank_idx] for d in decoded]

class BeamCTCDecoder(object):
	def __init__(self, labels, lm_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=100,
				 num_processes=4, blank_index=0):
		super(BeamCTCDecoder, self).__init__(labels)
		try:
			from ctcdecode import CTCBeamDecoder
		except ImportError:
			raise ImportError("BeamCTCDecoder requires paddledecoder package.")
		self._decoder = CTCBeamDecoder(labels.lower(), lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
									   num_processes, blank_index)

	def convert_to_strings(self, out, seq_len):
		results = []
		for b, batch in enumerate(out):
			utterances = []
			for p, utt in enumerate(batch):
				size = seq_len[b][p]
				transcript = ''.join(map(self.labels.idx2chr, utt[0:size])).upper() if size > 0 else ''
				utterances.append(transcript)
			results.append(utterances)
		return results

	def convert_tensor(self, offsets, sizes):
		results = []
		for b, batch in enumerate(offsets):
			utterances = []
			for p, utt in enumerate(batch):
				size = sizes[b][p]
				utterances.append(utt[0:size] if size > 0 else torch.tensor([], dtype=torch.int))
			results.append(utterances)
		return results

	def decode(self, probs, sizes=None):
		out, scores, offsets, seq_lens = self._decoder.decode(probs.permute(2, 0, 1).cpu(), sizes)
		return self.convert_to_strings(out, seq_lens), self.convert_tensor(offsets, seq_lens)

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

