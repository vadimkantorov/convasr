import shaping
import typing
from dataclasses import dataclass


@dataclass
class Segment:
	begin: float
	end: float
	text: str = None


@dataclass
class Transcript:
	segments: typing.List[Segment]

	@property
	def text(self):
		return ''.join(segment.text for segment in self.segments)

	def __iter__(self):
		return iter(self.segments)

	def __len__(self):
		return len(self.segments)


class GreedyCTCGenerator:
	def generate(self, tokenizer, log_probs: shaping.BCt, begin: float, end: float, output_lengths: typing.Optional[shaping.B] = None, time_stamps: typing.Optional[shaping.Bt] = None) -> typing.List[typing.List[Transcript]]:
		transcripts = []
		most_probable_idx = log_probs.argmax(dim = 1)
		for i in range(len(most_probable_idx)):
			sample_idx = most_probable_idx[i]
			sample_len = output_lengths[i] if output_lengths is not None else len(most_probable_idx[i])
			sample_ts = time_stamps[i] if time_stamps is not None else None
			transcript = Transcript(segments = [])

			t = 0
			while sample_idx[t] in tokenizer.silence_tokens and t < len(sample_idx):
				t += 1
			tokens = [tokenizer.eps_id]
			time_begin = begin + sample_ts[t] if sample_ts is not None else begin
			time_end = end

			for t in range(t, sample_len):
				if sample_idx[t] != tokenizer.eps_id and sample_idx[t] != tokens[-1]:
					tokens.append(sample_idx[t])
					time_end = begin + sample_ts[t] if sample_ts is not None else end
				if tokenizer.is_start_word_token(sample_idx[t]) and sample_ts is not None:
					segment = Segment(text = tokenizer.decode([tokens[1:]])[0], begin = time_begin, end = time_end)
					transcript.segments.append(segment)
					tokens = [tokenizer.eps_id]
					time_begin = begin + sample_ts[t] if sample_ts is not None else begin
					time_end = end

			if len(tokens) > 1:
				segment = Segment(text = tokenizer.decode([tokens[1:]])[0], begin = time_begin, end = time_end)
				transcript.segments.append(segment)
			transcripts.append([transcript])
		return transcripts
