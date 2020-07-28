import subprocess
import scipy.io.wavfile
import librosa
import torch
import models
import unittest
import time
import soundfile

f2s = lambda signal: (signal * torch.iinfo(torch.int16).max).short()
s2f = lambda signal: signal.float() / torch.iinfo(torch.int16).max


def read_audio(
	audio_path,
	sample_rate,
	offset = 0,
	duration = None,
	normalize = True,
	mono = True,
	byte_order = 'little',
	backend = 'ffmpeg',
	raw_s16le = None,
	raw_sample_rate = None,
	raw_num_channels = None,
	show_time = False
):
	start = time.time()
	try:
		if audio_path is None or audio_path.endswith('.raw'):
			if audio_path is not None:
				with open(audio_path, 'rb') as f:
					raw_s16le = f.read()
			sample_rate_, signal = raw_sample_rate, torch.ShortTensor(torch.ShortStorage.from_buffer(raw_s16le, byte_order = byte_order)).reshape(-1, raw_num_channels).t()
		elif audio_path.endswith('.wav'):
			sample_rate_, signal = scipy.io.wavfile.read(audio_path)
			signal = torch.as_tensor(signal[None, :] if len(signal.shape) == 1 else signal.T)
		elif audio_path.endswith('.gsm'):
			signal, sample_rate_ = soundfile.read(audio_path, dtype = 'float32')
			signal = torch.as_tensor(signal[None, :] if len(signal.shape) == 1 else signal.T)
		elif backend == 'sox':
			num_channels = int(subprocess.check_output(['soxi', '-V0', '-c', audio_path])) if not mono else 1
			params = [
				'sox',
				'-V0',
				audio_path,
				'-b',
				'16',
				'-e',
				'signed',
				'--endian',
				byte_order,
				'-r',
				str(sample_rate),
				'-c',
				str(num_channels),
				'-t',
				'raw',
				'-'
			]
			sample_rate_, signal = sample_rate, \
             torch.ShortTensor(torch.ShortStorage.from_buffer(subprocess.check_output(
				params), byte_order = byte_order)).reshape(-1, num_channels).t()
		elif backend == 'ffmpeg':
			num_channels = int(
				subprocess.check_output([
					'ffprobe',
					'-i',
					audio_path,
					'-show_entries',
					'stream=channels',
					'-select_streams',
					'a:0',
					'-of',
					'compact=p=0:nk=1',
					'-v',
					'0'
				])
			) if not mono else 1
			params = [
				'ffmpeg',
				'-i',
				audio_path,
				'-nostdin',
				'-hide_banner',
				'-nostats',
				'-loglevel',
				'quiet',
				'-f',
				's16le',
				'-ar',
				str(sample_rate),
				'-ac',
				str(num_channels),
				'-'
			]
			sample_rate_, signal = sample_rate, \
             torch.ShortTensor(torch.ShortStorage.from_buffer(subprocess.check_output(params), byte_order = byte_order)).reshape(-1, num_channels).t()
	except:
		print(f'Error when reading [{audio_path}]')
		sample_rate_, signal = sample_rate, torch.tensor([[]], dtype = torch.int16)

	assert signal.dtype in [torch.int16, torch.float32]
	if signal.dtype is torch.int16:
		signal = s2f(signal)
	if offset or duration is not None:
		signal = signal[...,
						slice(
							int(offset * sample_rate_) if offset else None,
							int((offset + duration) * sample_rate_) if duration is not None else None
						)]
	if mono:
		signal = signal.mean(dim = 0, keepdim = True)
	if normalize:
		signal = models.normalize_signal(signal, dim = -1)
	if sample_rate_ != sample_rate:
		signal, sample_rate_ = resample(signal, sample_rate_, sample_rate)

	if show_time:
		print('read audio time: ', time.time() - start)

	return signal, sample_rate_


def write_audio(audio_path, signal, sample_rate, mono = False):
	assert signal.dtype is torch.float32
	signal = signal if not mono else signal.mean(dim = 0, keepdim = True)
	scipy.io.wavfile.write(audio_path, sample_rate, f2s(signal.t()).numpy())
	return audio_path


def resample(signal, sample_rate_, sample_rate):
	return torch.from_numpy(librosa.resample(signal.numpy(), sample_rate_, sample_rate)), sample_rate


def compute_duration(audio_path, backend = 'ffmpeg'):
	cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1'
			] if backend == 'ffmpeg' else ['soxi', '-D'] if backend == 'sox' else None
	return float(subprocess.check_output(cmd + [audio_path]))


class AudioTests(unittest.TestCase):
	def test_should_read_gsm_file(self):
		audio_path = 'data/tests/audio.gsm'
		self.sample_rate = 8000
		self.mono = True
		self.audio_backend = 'ffmpeg'

		for i in range(100):
			signal, sample_rate = read_audio(audio_path, sample_rate = self.sample_rate, mono = self.mono, normalize = True, backend = self.audio_backend)
			assert sample_rate == 8000

	def test_should_read_amr_file(self):
		audio_path = 'data/tests/audio.amr'
		self.sample_rate = 8000
		self.mono = True
		self.audio_backend = 'ffmpeg'

		for i in range(100):
			signal, sample_rate = read_audio(audio_path, sample_rate = self.sample_rate, mono = self.mono, normalize = True, backend = self.audio_backend)
			assert sample_rate == 8000

	def test_should_read_wav_file(self):
		audio_path = 'data/tests/yg9FM5Zky2s.opus.0-99.900009-103.640007.wav'
		self.sample_rate = 8000
		self.mono = True
		self.audio_backend = 'ffmpeg'

		for i in range(100):
			signal, sample_rate = read_audio(audio_path, sample_rate = self.sample_rate, mono = self.mono, normalize = True, backend = self.audio_backend, show_time=False)
			assert sample_rate == 8000

	def test_should_read_opus_file(self):
		audio_path = 'data/tests/audio.opus'
		self.sample_rate = 8000
		self.mono = True
		self.audio_backend = 'ffmpeg'

		for i in range(100):
			signal, sample_rate = read_audio(audio_path, sample_rate = self.sample_rate, mono = self.mono, normalize = True, backend = self.audio_backend, show_time=False)
			assert sample_rate == 8000


if __name__ == '__main__':
	audio_path = 'data/tests/audio.wav'
	sample_rate = 8000
	mono = True
	audio_backend = 'ffmpeg'

	for i in range(100):
		signal, sample_rate = read_audio(audio_path, sample_rate=sample_rate, mono=mono, normalize=True,
			backend=audio_backend)
		assert sample_rate == 8000
