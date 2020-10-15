import shaping
import typing
import transcripts

class GreedyCTCGenerator:
	def generate(self,
	             tokenizer,
	             log_probs: shaping.BCt,
	             begin: shaping.B,
	             end: shaping.B,
	             output_lengths: typing.Optional[shaping.B] = None,
	             time_stamps: typing.Optional[shaping.Bt] = None,
	             segment_text_key: str = 'hyp',
	             segment_extra_info: typing.List[dict] = None) -> typing.List[typing.List[transcripts.Transcript]]:
		_transcripts = []
		most_probable_idx = log_probs.argmax(dim = 1)
		for i in range(len(most_probable_idx)):
			sample_idx = most_probable_idx[i]
			sample_len = output_lengths[i] if output_lengths is not None else len(most_probable_idx[i])
			sample_ts = time_stamps[i] if time_stamps is not None else None
			transcript = transcripts.Transcript()

			t = 0
			while sample_idx[t] in tokenizer.silence_tokens and t < len(sample_idx):
				t += 1
			tokens = [tokenizer.eps_id]
			time_begin = begin[i] + sample_ts[t] if sample_ts is not None else begin[i]
			time_end = end[i]

			for t in range(t, sample_len):
				if sample_idx[t] == tokenizer.eps_id or sample_idx[t] == tokens[-1]:
					continue
				if tokenizer.is_start_word_token(sample_idx[t]) and sample_ts is not None:
					segment = transcripts.Segment(begin = time_begin.item(),
					                              end = time_end.item(),
					                              **{segment_text_key: tokenizer.decode([tokens[1:]])[0]})
					if segment_extra_info is not None:
						segment.update(segment_extra_info[i])
					transcript.append(segment)
					tokens = [tokenizer.eps_id, sample_idx[t]]
					time_begin = begin[i] + sample_ts[t] if sample_ts is not None else begin[i]

				tokens.append(sample_idx[t])
				time_end = begin[i] + sample_ts[t] if sample_ts is not None else end[i]

			if len(tokens) > 1:
				segment = transcripts.Segment(begin = time_begin.item(),
				                              end = time_end.item(),
				                              **{segment_text_key: tokenizer.decode([tokens[1:]])[0]})
				if segment_extra_info is not None:
					segment.update(segment_extra_info[i])
				transcript.append(segment)
			_transcripts.append([transcript])
		return _transcripts
