import subprocess
import time
import torch
import numpy as np
import librosa
import soundfile
import scipy.io.wavfile
import models

smax = torch.iinfo(torch.int16).max
f2s_numpy = lambda signal, max = np.float32(smax): np.multiply(signal, max, dtype = 'int16')
s2f_numpy = lambda signal, max = np.float32(smax): np.divide(signal, max, dtype = 'float32')


def read_audio(
	audio_path,
	sample_rate,
	offset = 0,
	duration = None,
	normalize = True,
	mono = True,
	byte_order = 'little',
	backend = None,
	raw_s16le = None,
	raw_sample_rate = None,
	raw_num_channels = None,
):
	try:
		if audio_path is None or audio_path.endswith('.raw'):
			if audio_path is not None:
				with open(audio_path, 'rb') as f:
					raw_s16le = f.read()
			sample_rate_, signal = raw_sample_rate, np.frombuffer(raw_s16le, dtype = 'int16').reshape(-1, raw_num_channels).T

		elif backend in ['scipy', None] and audio_path.endswith('.wav'):
			sample_rate_, signal = scipy.io.wavfile.read(audio_path)
			signal = signal[None, :] if len(signal.shape) == 1 else signal.T

		elif backend == 'soundfile':
			signal, sample_rate_ = soundfile.read(audio_path, dtype = 'int16')
			signal = signal[None, :] if len(signal.shape) == 1 else signal.T

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
			sample_rate_, signal = sample_rate, np.frombuffer(subprocess.check_output(params), dtype = 'int16').reshape(-1, num_channels).T
		elif backend in ['ffmpeg', None]:
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
			sample_rate_, signal = sample_rate, np.frombuffer(subprocess.check_output(params), dtype = 'int16').reshape(-1, num_channels).T

	except:
		print(f'Error when reading [{audio_path}]')
		sample_rate_, signal = sample_rate, np.array([[]], dtype = 'int16')

	if offset or duration is not None:
		signal = signal[...,
						slice(
							int(offset * sample_rate_) if offset else None,
							int((offset + duration) * sample_rate_) if duration is not None else None
						)]
	
	assert 'int16' in str(signal.dtype) or 'float32' in str(signal.dtype) 
	if 'int16' in str(signal.dtype):
		signal = s2f_numpy(signal)
	if mono and len(signal) > 1:
		signal = signal.mean(0, keepdims = True)

	signal = torch.as_tensor(signal)
	if normalize:
		signal = models.normalize_signal(signal, dim = -1)
	if sample_rate_ != sample_rate:
		signal, sample_rate_ = resample(signal, sample_rate_, sample_rate)

	return signal, sample_rate_


def write_audio(audio_path, signal, sample_rate, mono = False):
	assert signal.dtype is torch.float32
	signal = signal if not mono else signal.mean(dim = 0, keepdim = True)
	scipy.io.wavfile.write(audio_path, sample_rate, f2s_numpy(signal.t().numpy()))
	return audio_path


def resample(signal, sample_rate_, sample_rate):
	return torch.from_numpy(librosa.resample(signal.numpy(), sample_rate_, sample_rate)), sample_rate


def compute_duration(audio_path, backend = 'ffmpeg'):
	cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1'
			] if backend == 'ffmpeg' else ['soxi', '-D'] if backend == 'sox' else None
	return float(subprocess.check_output(cmd + [audio_path]))


def timeit(audio_path, number, sample_rate, mono, audio_backend, scale):
	for i in range(3):
		read_audio(audio_path, sample_rate = sample_rate, mono = mono, backend = audio_backend, normalize=False)
	
	start_process_time = time.process_time_ns()
	start_perf_counter = time.perf_counter_ns()
	for i in range(number):
		read_audio(audio_path, sample_rate = sample_rate, mono = mono, backend = audio_backend, normalize=False)
	end_process_time = time.process_time_ns()
	end_perf_counter = time.perf_counter_ns()
	process_time = (end_process_time - start_process_time) / scale / number
	perf_counter = (end_perf_counter - start_perf_counter) / scale / number

	print(f'|{audio_path:>20}|{number:>5}|{audio_backend:>10}|{process_time:9.0f}|{perf_counter:9.0f}|')


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()
	cmd = subparsers.add_parser('timeit')

	cmd.add_argument('--audio-path', type = str, required = True)
	cmd.add_argument('--sample-rate', type = int, default = 8000)
	cmd.add_argument('--mono', action = 'store_true')
	cmd.add_argument('--audio-backend', type = str, required = True)
	cmd.add_argument('--number', type = int, default = 100)
	cmd.add_argument('--scale', type = int, default = 1000)
	cmd.set_defaults(func = timeit)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
