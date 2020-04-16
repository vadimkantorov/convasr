import subprocess
import scipy.io.wavfile
import librosa
import torch
import models

f2s = lambda signal: (signal * torch.iinfo(torch.int16).max).short()
s2f = lambda signal: signal.float() * torch.iinfo(torch.int16).max

def read_audio(audio_path, sample_rate, offset = 0, duration = None, normalize = True, mono = True, byte_order = 'little', backend = 'sox'):
	try:
		if audio_path.endswith('.wav'):
			sample_rate_, signal = scipy.io.wavfile.read(audio_path)
			signal = torch.as_tensor(signal[None, :] if len(signal.shape) == 1 else signal.T)
		elif backend == 'sox':
			num_channels = int(subprocess.check_output(['soxi', '-V0', '-c', audio_path])) if not mono else 1
			sample_rate_, signal = sample_rate, torch.ShortTensor(torch.ShortStorage.from_buffer(subprocess.check_output(['sox', '-V0', audio_path, '-b', '16', '-e', 'signed', '--endian', byte_order, '-r', str(sample_rate), '-c', str(num_channels), '-t', 'raw', '-']), byte_order = byte_order)).reshape(-1, num_channels).t()
		elif backend == 'ffmpeg':
			num_channels = int(subprocess.check_output(['ffprobe', '-i', audio_path, '-show_entries', 'stream=channels', '-select_streams', 'a:0', '-of', 'compact=p=0:nk=1', '-v', '0'])) if not mono else 1
			sample_rate_, signal = sample_rate, torch.ShortTensor(torch.ShortStorage.from_buffer(subprocess.check_output(['ffmpeg', '-i', audio_path, '-nostdin', '-hide_banner', '-nostats', '-loglevel', 'quiet', '-f', 's16le', '-ar', str(sample_rate), '-ac', str(num_channels), '-']), byte_order = byte_order)).reshape(-1, num_channels).t()
	except:
		print(f'Error when reading [{audio_path}]')
		sample_rate_, signal = sample_rate, torch.tensor([[]], dtype = dtype)

	assert signal.dtype in [torch.int16, torch.float32]
	if signal.dtype is torch.int16:
		signal = s2f(signal)
	if offset or duration is not None:
		signal = signal[..., slice(int(offset * sample_rate_) if offset else None, int((offset + duration) * sample_rate_) if duration is not None else None)]
	if mono:
		signal = signal.mean(dim = 0, keepdim = True)
	if normalize:
		signal = models.normalize_signal(signal, dim = -1)
	if sample_rate_ != sample_rate:
		signal, sample_rate_ = resample(signal, sample_rate_, sample_rate)

	return signal, sample_rate_

def write_audio(audio_path, signal, sample_rate, mono = False):
	assert signal.dtype is torch.float32
	signal = signal if not mono else signal.mean(dim = 0, keepdim = True)
	scipy.io.wavfile.write(audio_path, sample_rate, f2s(signal.t()).numpy())
	return audio_path

def resample(signal, sample_rate_, sample_rate):
	return torch.from_numpy(librosa.resample(signal.numpy(), sample_rate_, sample_rate)), sample_rate
