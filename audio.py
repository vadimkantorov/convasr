import librosa
import torch
import models

def read_audio(audio_path, sample_rate, normalize = True, mono = True, duration = None, dtype = torch.float32, byte_order = 'little'):
	if audio_path.endswith('.wav'):
		sample_rate_, signal = scipy.io.wavfile.read(audio_path) 
		signal = signal[None, :] if len(signal.shape) == 1 else signal.T
	else:
		num_channels = int(subprocess.check_output(['soxi', '-V0', '-c', audio_path])) if not mono else 1
		sample_rate_, signal = sample_rate, torch.ShortTensor(torch.ShortStorage.from_buffer(subprocess.check_output(['sox', '-V0', audio_path, '-b', '16', '-e', 'signed', '--endian', byte_order, '-r', str(sample_rate), '-c', str(num_channels), '-t', 'raw', '-']), byte_order = byte_order)).reshape(-1, num_channels).t()

	signal = torch.as_tensor(signal).to(dtype)
	
	if duration is not None:
		signal = signal[:int(duration * sample_rate_), ...]
	if mono:
		signal = signal.float().mean(dim = 0, keepdim = True).to(dtype)
	if normalize:
		signal = models.normalize_signal(signal, dim = -1)
	if dtype is torch.float32 and sample_rate_ != sample_rate:
		signal, sample_rate_ = resample(signal, sample_rate_, sample_rate)

	assert sample_rate_ == sample_rate, 'Cannot resample non-float tensors because of librosa constraints'
	return signal, sample_rate_

def write_audio(audio_path, signal, sample_rate):
	scipy.io.wavfile.write(audio_path, sample_rate, signal.t().numpy())
	return audio_path

def resample(signal, sample_rate_, sample_rate):
	return torch.from_numpy(librosa.resample(signal.numpy(), sample_rate_, sample_rate)), sample_rate
