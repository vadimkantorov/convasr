import subprocess
import torch
import numpy as np
import librosa
import soundfile
import scipy.io.wavfile

smax = torch.iinfo(torch.int16).max
f2s_numpy = lambda signal, max = np.float32(smax): np.multiply(signal, max).astype('int16')
s2f_numpy = lambda signal, max = np.float32(smax): np.divide(signal, max, dtype = 'float32')

def read_audio(
        audio_path,
        sample_rate,
        offset=0,
        duration=None,
        mono=True,
        raw_dtype='int16',
        dtype='float32',
        byte_order='little',
        backend=None,
        raw_bytes=None,
        raw_sample_rate=None,
        raw_num_channels=None,
):
	assert dtype in ['int16', 'float32']

	try:
		if audio_path is None or audio_path.endswith('.raw'):
			if audio_path is not None:
				with open(audio_path, 'rb') as f:
					raw_bytes = f.read()
			sample_rate_, signal = raw_sample_rate, np.frombuffer(raw_bytes, dtype = raw_dtype).reshape(-1, raw_num_channels)

		elif backend in ['scipy', None] and audio_path.endswith('.wav'):
			sample_rate_, signal = scipy.io.wavfile.read(audio_path)
			signal = signal[:, None] if len(signal.shape) == 1 else signal

		elif backend == 'soundfile':
			signal, sample_rate_ = soundfile.read(audio_path, dtype = raw_dtype)
			signal = signal[:, None] if len(signal.shape) == 1 else signal

		elif backend == 'sox':
			num_channels = int(subprocess.check_output(['soxi', '-V0', '-c', audio_path])) if not mono else 1
			params_fmt = ['-b', '16', '-e', 'signed'] if raw_dtype == 'int16' else ['-b', '32', '-e', 'float']
			params = [
				'sox',
				'-V0',
				audio_path
				] + params_fmt +[
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
			sample_rate_, signal = sample_rate, np.frombuffer(subprocess.check_output(params), dtype = raw_dtype).reshape(-1, num_channels)
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
			params_fmt = ['-f', 's16le'] if raw_dtype == 'int16' else ['-f', 'f32le']
			params = [
				'ffmpeg',
				'-i',
				audio_path,
				'-nostdin',
				'-hide_banner',
				'-nostats',
				'-loglevel',
				'quiet'] + params_fmt + [
				'-ar',
				str(sample_rate),
				'-ac',
				str(num_channels),
				'-'
			]
			sample_rate_, signal = sample_rate, np.frombuffer(subprocess.check_output(params), dtype = raw_dtype).reshape(-1, num_channels)

	except:
		raise
		print(f'Error when reading [{audio_path}]')
		sample_rate_, signal = sample_rate, np.array([[]], dtype = dtype)

	if offset or duration is not None:
		signal = signal[
						slice(
							int(offset * sample_rate_) if offset else None,
							int((offset + duration) * sample_rate_) if duration is not None else None
						)]
	
	assert signal.dtype in [np.int16, np.float32]
	signal = signal.T
	
	if signal.dtype == np.int16 and dtype == 'float32':
		signal = s2f_numpy(signal)
	
	if mono and len(signal) > 1:
		assert signal.dtype == np.float32
		signal = signal.mean(0, keepdims = True)

	signal = torch.as_tensor(signal)

	if sample_rate_ != sample_rate:
		signal, sample_rate_ = resample(signal, sample_rate_, sample_rate)

	return signal, sample_rate_


def write_audio(audio_path, signal, sample_rate, mono = False):
	assert signal.dtype is torch.float32
	signal = signal if not mono else signal.mean(dim = 0, keepdim = True)
	scipy.io.wavfile.write(audio_path, sample_rate, f2s_numpy(signal.t().numpy()))
	return audio_path


def resample(signal, sample_rate_, sample_rate):
	assert signal.dtype == torch.float32
	mono = len(signal) == 1
	if mono:
		signal = signal.squeeze(0)
	# librosa does not like mono 1T signals
	signal = torch.as_tensor(librosa.resample(signal.numpy(), sample_rate_, sample_rate))
	if mono:
		signal = signal.unsqueeze(0)
	return signal, sample_rate

def compute_duration(audio_path, backend='ffmpeg'):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1'
           ] if backend == 'ffmpeg' else ['soxi', '-D'] if backend == 'sox' else None
    return float(subprocess.check_output(cmd + [audio_path]))


if __name__ == '__main__':
    import argparse
    import time
    import utils

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    cmd = subparsers.add_parser('timeit')

    cmd.add_argument('--audio-path', type=str, required=True)
    cmd.add_argument('--sample-rate', type=int, default=8000)
    cmd.add_argument('--mono', action='store_true')
    cmd.add_argument('--audio-backend', type=str, required=True)
    cmd.add_argument('--number', type=int, default=100)
    cmd.add_argument('--number-warmup', type=int, default=3)
    cmd.add_argument('--scale', type=int, default=1000)
    cmd.add_argument('--raw-dtype', default='int16', choices=['int16', 'float32'])
    cmd.add_argument('--dtype', default='float32', choices=['int16', 'float32'])
    cmd.set_defaults(func='timeit')

    args = parser.parse_args()

    if args.func == 'timeit':
        utils.reset_cpu_threads(1)
        for i in range(args.number_warmup):
            read_audio(args.audio_path, sample_rate=args.sample_rate, mono=args.mono, backend=args.audio_backend, dtype=args.dtype, raw_dtype=args.raw_dtype)

        start_process_time = time.process_time_ns()
        start_perf_counter = time.perf_counter_ns()
        for i in range(args.number):
            read_audio(args.audio_path, sample_rate=args.sample_rate, mono=args.mono, backend=args.audio_backend, dtype=args.dtype, raw_dtype=args.raw_dtype)
        end_process_time = time.process_time_ns()
        end_perf_counter = time.perf_counter_ns()
        process_time = (end_process_time - start_process_time) / args.scale / args.number
        perf_counter = (end_perf_counter - start_perf_counter) / args.scale / args.number
        print(f'|{args.audio_path:>20}|{args.number:>5}|{args.audio_backend:>10}|{process_time:9.0f}|{perf_counter:9.0f}|')
