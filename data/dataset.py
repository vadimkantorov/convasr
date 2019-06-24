import numpy as np
import scipy.io.wavfile
import torch.utils.data

def read_wav(path, channel=-1):
	sample_rate, signal = scipy.io.wavfile.read(path)
	signal = torch.from_numpy(signal)
	abs_max = signal.abs().max()
	if abs_max > 0:
		signal *= 1. / abs_max
    
	if len(signal.shape) > 1:
		if signal.shape[1] == 1:
			signal = signal.squeeze()
		elif channel == -1:
			signal = signal.mean(1)  # multiple channels, average
		else:
			signal = signal[:, channel]  # multiple channels, average
	return signal, sample_rate

class SpectrogramDataset(torch.utils.data.Dataset):
	def __init__(self):
		pass

	def __getitem__(self, idx):
		pass

	def __len__(self):
		pass
