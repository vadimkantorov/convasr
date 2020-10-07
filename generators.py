import shaping
import typing


class GreedyGenerator:
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer

	@property
	def name(self):
		return f'greedy_{self.tokenizer.name}'

	def generate(self, log_probs: shaping.BCt, begin: float, end: float, output_lengths: shaping.B = None, time_stamps: shaping.Bt = None) -> typing.List[typing.List[dict]]:
		hypotheses = []
		most_probable_idx = log_probs.argmax(dim = 1)
		for i in range(len(most_probable_idx)):
			sample_tokens = most_probable_idx[i]
			sample_time_stamps = time_stamps[i] if time_stamps is not None else None
			if output_lengths is not None:
				sample_tokens = sample_tokens[:output_lengths[i]]
				sample_time_stamps = sample_time_stamps[:output_lengths[i]] if sample_time_stamps is not None else None

			if sample_time_stamps is None:
				hypotheses.append([[dict(hyp = self.tokenizer.decode([sample_tokens])[0], begin = begin, end = end)]])
			else:
				hypothesis = []
				t = 0
				segment = dict(hyp = [], begin = None, end = None)
				while sample_tokens[t] in self.tokenizer.silence_tokens and t < len(sample_tokens):
					t += 1
					segment['hyp'].append(sample_tokens[t])
				segment['begin'] = begin + sample_time_stamps[t]
				for t in range(t, len(sample_tokens)):
					if self.tokenizer.is_start_word_token(sample_tokens[t]):
						segment['hyp'] = self.tokenizer.decode([segment['hyp']])[0]
						hypothesis.append(segment)
						segment = dict(hyp = [], begin = None, end = None)
					if sample_tokens[t] not in self.tokenizer.silence_tokens:
						if segment['begin'] is None:
							segment['begin'] = begin + sample_time_stamps[t]
						else:
							segment['end'] = begin + sample_time_stamps[t]
					segment['hyp'] += sample_tokens[t]
				hypotheses.append(hypothesis)
		return hypotheses
