import torch
import shaping

class GreedyDecoder:
	def decode(self, log_probs : shaping.BCt, output_lengths : typing.Optional[shaping.B] = None, K = 1):
		# returns list of lists B x l    # TODO: (B x H x l) 
		return [
			l[... if K > 1 else 0, :o].tolist() for o,
			l in zip(
				torch
				.as_tensor(output_lengths if output_lengths is not None else [log_probs.shape[-1]] *
							len(log_probs)).tolist(),
				log_probs.topk(K, dim = 1).indices
			)
		]


class BeamSearchDecoder:
	def __init__(
		self,
		labels,
		lm_path,
		beam_width,
		beam_alpha = 0,
		beam_beta = 0,
		cutoff_top_n = 40,
		cutoff_prob = 1.0,
		num_workers = 1,
		topk = 1
	):
		import ctcdecode
		self.topk = topk
		self.beam_search_decoder = ctcdecode.CTCBeamDecoder(
			list(str(labels).lower()),
			lm_path,
			beam_alpha,
			beam_beta,
			cutoff_top_n if cutoff_top_n is not None else len(labels),
			cutoff_prob,
			beam_width,
			num_workers,
			labels.blank_idx,
			log_probs_input = True
		)

	def decode(self, log_probs, output_lengths):
		list_or_one = lambda xs: xs if len(xs) > 1 else xs[0]
		decoded_chr, decoded_scores, decoded_offsets, decoded_lengths = self.beam_search_decoder.decode(log_probs.permute(0, 2, 1).cpu(), torch.as_tensor(output_lengths).cpu().int())
		decoded_top_scores, decoded_top_inds = decoded_scores.topk(self.topk, dim = 1)
		return [
			list_or_one([d[t_, :l[t_]].tolist() for t_ in t.tolist()]) for d,
			l,
			t in zip(decoded_chr, decoded_lengths, decoded_top_inds)
		]  #, [list_or_one(t) for t in decoded_top_scores.tolist()]
