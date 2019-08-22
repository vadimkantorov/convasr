import torch

class GreedyDecoder(object):
	def __init__(self, labels):
	   self.labels = labels 

	def decode(self, log_probs, output_lengths, K = 1):
		decoded_idx = log_probs.argmax(dim = 1).tolist()
		return [[i for k, i in enumerate(d) if (k == 0 or i != d[k - 1]) and i != self.labels.blank_idx] for d in decoded_idx]

class BeamSearchDecoder(object):
	def __init__(self, labels, lm_path, beam_width, beam_alpha = 0, beam_beta = 0, cutoff_top_n = 40, cutoff_prob = 1.0, num_workers = 1, topk = 1):
		import ctcdecode
		self.topk = topk
		self.beam_search_decoder = ctcdecode.CTCBeamDecoder(str(labels).lower(), lm_path, beam_alpha, beam_beta, cutoff_top_n if cutoff_top_n is not None else len(labels), cutoff_prob, beam_width, num_workers, labels.blank_idx, log_probs_input = True)

	def decode(self, log_probs, output_lengths):
		list_or_one = lambda xs: xs if len(xs) > 1 else xs[0]
		decoded_chr, decoded_scores, decoded_offsets, decoded_lengths = self.beam_search_decoder.decode(log_probs.permute(0, 2, 1).cpu(), torch.IntTensor(output_lengths))
		decoded_top = decoded_scores.topk(self.topk, dim = 1).indices
		return [list_or_one([d[int(t_)][:l[int(t_)]].tolist() for t_ in t.tolist()]) for d, l, t in zip(decoded_chr, decoded_lengths, decoded_top)]
