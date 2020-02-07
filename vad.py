import torch
import torch.nn.functional as F
import webrtcvad

#goals:
# augmenting decoding
# batching
# cuts for model-less dataset creation
# diarization?

def detect_speech(signal, sample_rate, window_size, aggressiveness, postproc = lambda speech: speech):
	vad = webrtcvad.Vad(aggressiveness)
	frame_len = int(window_size * sample_rate)
	speech = torch.as_tensor([[len(chunk) == frame_len and vad.is_speech(bytearray(chunk.numpy()), sample_rate) for chunk in channel.split(frame_len)] for channel in signal])
	speech = postproc(speech)
	return speech.repeat_interleave(frame_len, dim = -1)[:, :signal.shape[1]]

def postprocess_cut(speech):
	#  expand a bit, 
	
	# 1. merge if gap < 1 sec
	# 2. remove if len < 0.5 sec
	# 3. filter by energy
	# 4. cut sends by energy
	
	pass

def postprocess_batching(speech):
	# expand a lot
	# cut by max length
	pass

def postprocess_decoding(speech):
	kernel_size = 101
	return F.max_pool1d(speech.unsqueeze(1).float(), stride = 1, kernel_size = kernel_size, padding = kernel_size // 2).squeeze(1).to(speech.dtype)
