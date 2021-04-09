import torch
import typing

import shaping
import transcripts


class GreedyCTCGenerator:
	def __init__(self, blank_amount_to_space: int = 10):
		"""
		Args:
			blank_amount_to_space: how many blank symbols to transform to single space
		"""
		self.blank_amount_to_space = blank_amount_to_space

	def generate(
			self,
			tokenizer,
			log_probs: shaping.BCt,
			begin: shaping.B,
			end: shaping.B,
			output_lengths: typing.Optional[shaping.B] = None,
			time_stamps: typing.Optional[shaping.Bt] = None,
			segment_text_key: str = 'hyp',
			segment_extra_info: typing.List[dict] = None) -> typing.List[typing.List[transcripts.Transcript]]:
		_transcripts = []
		most_probable_idx = log_probs.argmax(dim = 1).cpu().tolist()
		time_stamps = time_stamps.cpu().tolist() if time_stamps is not None else None
		begin = torch.clamp(begin, min = 0.0).cpu().tolist() if time_stamps is not None else begin.cpu().tolist()
		end = end.cpu().tolist()

		for i in range(len(most_probable_idx)):
			sample_idx = most_probable_idx[i]
			sample_len = output_lengths[i] if output_lengths is not None else len(most_probable_idx[i])
			sample_ts = time_stamps[i] if time_stamps is not None else None
			transcript = transcripts.Transcript()

			t = 0
			while t < len(sample_idx) and sample_idx[t] in tokenizer.silence_tokens_ids:
				t += 1
			if t >= len(sample_idx):
				_transcripts.append([transcript])
				continue
			tokens = [tokenizer.eps_id]
			time_begin = begin[i] + sample_ts[t] if sample_ts is not None else begin[i]
			time_end = end[i]

			allow_tokens_repeat = False
			count_eps_id = 0

			for t in range(t, sample_len):
				if sample_idx[t] == tokenizer.eps_id and tokens[-1] == tokenizer.space_id:
					continue
				if sample_idx[t] == tokenizer.eps_id:
					allow_tokens_repeat = True
					count_eps_id += 1

					# try to add space
					if count_eps_id >= self.blank_amount_to_space and not tokenizer.is_start_word_token(tokens[-1]):
						tokens.append(tokenizer.space_id)

					continue

				elif sample_idx[t] == tokens[-1] and not allow_tokens_repeat:
					# TODO here we skip cases with double consonant letter ?
					continue

				if tokenizer.is_start_word_token(sample_idx[t]) and sample_ts is not None:
					segment = transcripts.Segment(begin=time_begin,
												  end=time_end,
												  **{segment_text_key: tokenizer.decode([tokens[1:]])[0]})

					if segment_extra_info is not None:
						segment.update(segment_extra_info[i])

					transcript.append(segment)
					tokens = [tokenizer.eps_id, sample_idx[t]]
					time_begin = begin[i] + sample_ts[t] if sample_ts is not None else begin[i]

				allow_tokens_repeat = False
				tokens.append(sample_idx[t])
				time_end = begin[i] + sample_ts[t] if sample_ts is not None else end[i]
				count_eps_id = 0

			if len(tokens) > 1:
				segment = transcripts.Segment(begin=time_begin,
											  end=time_end,
											  **{segment_text_key: tokenizer.decode([tokens[1:]])[0]})
				if segment_extra_info is not None:
					segment.update(segment_extra_info[i])
				transcript.append(segment)
			_transcripts.append([transcript])
		return _transcripts
