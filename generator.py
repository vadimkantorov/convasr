import shaping


class GreedyGenerator:
	def decode(self, log_probs: shaping.BCt, output_lengths: shaping.B = None):
		tokens = []
		most_probable_idx = log_probs.argmax(dim = 1)
		for i in range(len(most_probable_idx)):
			sample_tokens = most_probable_idx[i].tolist()
			if output_lengths is not None:
				sample_tokens = sample_tokens[:output_lengths[i]]
			tokens.append([sample_tokens])
		return tokens
