import torch
import torch.nn.functional as F
import webrtcvad

#goals:
# augmenting decoding
# batching
# cuts for dataset creation
# diarization?

def detect_speech(signal, sample_rate, window_size, aggressiveness):
	vad = webrtcvad.Vad(aggressiveness)
	frame_len = int(window_size * sample_rate)
	return torch.as_tensor([[len(chunk) == frame_len and vad.is_speech(bytearray(chunk.numpy()), sample_rate) for chunk in channel.split(frame_len)] for channel in signal.t()]).t().repeat_interleave(frame_len, dim = 0)[:len(signal)]
	
# does not filter anything out, can only merge
def segment(speech, max_duration = None):
	_notspeech_ = ~F.pad(speech, [1, 1])
	(begin,), (end,) = (speech & _notspeech_[:-2]).nonzero(as_tuple = True), (speech & _notspeech_[2:]).nonzero(as_tuple = True)
	
	#sec = lambda k: k / len(idx) * (e - b)
	#i = 0
	#for j in range(1, 1 + len(idx)):
	#	if j == len(idx) or (idx[j] == labels.space_idx and sec(j - 1) - sec(i) > max_segment_seconds):
	#		yield (b + sec(i), b + sec(j - 1), labels.postprocess_transcript(labels.decode(idx[i:j])[0]))
	#		i = j + 1
	
	return [dict(i = i, j = j) for i, j in zip(begin.tolist(), end.tolist())]

	#begin_end_ = ((frame_len * torch.IntTensor(begin_end)).float() / sample_rate).tolist()
	
	# 1. merge if gap < 1 sec
	# 2. remove if len < 0.5 sec
	# 3. filter by energy
	# 4. cut sends by energy
	
