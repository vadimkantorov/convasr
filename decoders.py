import torch

class GreedyDecoder:
	def decode(self, log_probs, output_lengths, K = 1):
		return [l[:o] for o, l in zip(torch.as_tensor(output_lengths).tolist(), log_probs.argmax(dim = 1).tolist())]#, [1.0] * len(log_probs)

class GreedyNoBlankDecoder:
	def decode(self, log_probs, output_lengths, K = 1):
		def decode_(argmax_eps, argmax_eps_):
			space_idx = log_probs.shape[1] - 2
			space = True
			res = []
			for i, i_ in zip(argmax_eps, argmax_eps_):
				j = i if space or i_ == space_idx else i_
				res.append(j)
				if j == space_idx:
					space = True
				elif j < space_idx:
					space = False
			return res

		return [decode_(l[:o], l_[:o]) for o, l, l_ in zip(torch.as_tensor(output_lengths).tolist(), log_probs.argmax(dim = 1).tolist(), log_probs[:, :-1, :].argmax(dim = 1).tolist())]

class BeamSearchDecoder:
	def __init__(self, labels, lm_path, beam_width, beam_alpha = 0, beam_beta = 0, cutoff_top_n = 40, cutoff_prob = 1.0, num_workers = 1, topk = 1):
		import ctcdecode
		self.topk = topk
		self.beam_search_decoder = ctcdecode.CTCBeamDecoder(str(labels).lower(), lm_path, beam_alpha, beam_beta, cutoff_top_n if cutoff_top_n is not None else len(labels), cutoff_prob, beam_width, num_workers, labels.blank_idx, log_probs_input = True)

	def decode(self, log_probs, output_lengths):
		list_or_one = lambda xs: xs if len(xs) > 1 else xs[0]
		decoded_chr, decoded_scores, decoded_offsets, decoded_lengths = self.beam_search_decoder.decode(log_probs.permute(0, 2, 1).cpu(), torch.as_tensor(output_lengths).cpu().int())
		decoded_top_scores, decoded_top_inds = decoded_scores.topk(self.topk, dim = 1)
		return [list_or_one([d[t_, :l[t_]].tolist() for t_ in t.tolist()]) for d, l, t in zip(decoded_chr, decoded_lengths, decoded_top_inds)]#, [list_or_one(t) for t in decoded_top_scores.tolist()]
