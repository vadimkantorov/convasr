import numpy as np
import torch.utils.data

import scipy.io.wavfile
import librosa

class SpectrogramDataset(torch.utils.data.Dataset):
	def __init__(self, audio_conf, data_path, labels_path):
		self.window_stride = audio_conf['window_stride']
		self.window_size = audio_conf['window_size']
		self.sample_rate = audio_conf['sample_rate']
		self.window = getattr(scipy.signal, audio_conf.get('window', 'hamming'))
 		self.labels = Labels(labels)
		self.transforms = []

	def __getitem__(self, index):
		audio_path, transcript, duration = self.ids[index]
		
		signal, sample_rate = read_wav(audio_path); 
		if sample_rate != self.sample_rate:
			signal, sample_rate = librosa.resample(y, sample_rate, self.sample_rate), self.sample_rate
		# apply self.transforms 
		spect = spectrogram(signal, sample_rate, self.window_size, self.window_stride, self.window)
		transcript = self.labels.parse(transcrip)
		return spect, transcript, audio_path

	def __len__(self):
		pass

def spectrogram(signal, sample_rate, window_size, window_stride, window):
	n_fft = int(sample_rate * (window_size + 1e-8))
	win_length = n_fft
	hop_length = int(sample_rate * (window_stride + 1e-8))
	D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
	spect = np.abs(D)

	if spect.shape[0] < 161:
		spect.resize((161, *spect.shape[1:]))
		spect[81:] = spect[80:0:-1]
	spect = spect[:161]
	return spect

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
		assert len(signal.shape) == 1
	return signal, sample_rate
