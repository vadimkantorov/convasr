import os
import tempfile
import random
import subprocess
import torch
import dataset
import librosa
import torchaudio
import models
import numpy as np

fixed_or_uniform = lambda r: random.uniform(*r) if isinstance(r, list) else r

class RandomCompose(object):
	def __init__(self, transforms, prob):
		self.transforms = transforms
		self.prob = prob

	def __call__(self, *x):
		if random.random() < self.prob:
			transform = random.choice(self.transforms)
			x = transform(*x)
		return x

	def __str__(self):
		return self.__class__.__name__.replace('RandomCompose', '') + '_'.join(t.__class__.__name__ + ('__'.join(str(y) for x in t.__dict__.values() for y in (x if isinstance(x, list) else [x]))) for t in self.transforms)

class RandomComposeSox(RandomCompose):
	def __init__(self, transforms, prob):
		super().__init__(transforms, prob)
		self.initialize_sox = True

	def __call__(self, audio_path, sample_rate, normalize = True):
		if self.initialize_sox:
			#torchaudio.initialize_sox()
			self.initialize_sox = False
		effect = None
		if self.transforms and random.random() < self.prob:
			transform = random.choice(self.transforms)
			effect = ['pitch', fixed_or_uniform(transform.n_steps) * 100] if isinstance(transform, PitchShift) else ['tempo', fixed_or_uniform(transform.rate)] if isinstance(transform, SpeedPerturbation) else ['gain', fixed_or_uniform(transform.gain_db)] if isinstance(transform, GainPerturbation) else 'transcode' if isinstance(transform, Transcode) else []

		tmp_audio_path = []
		if effect == 'transcode':
			tmp_audio_path = [tempfile.mkstemp(suffix = '.' + transform.codec, dir = transform.tmpdir)[1], tempfile.mkstemp(suffix = '.wav', dir = transform.tmpdir)[1]]
			subprocess.check_call(['sox', '-V0', audio_path, '-t', transform.codec, '-r', str(transform.sample_rate), tmp_audio_path[0]])
			subprocess.check_call(['sox', '-V0', tmp_audio_path[0], '-t', 'wav', tmp_audio_path[1]])
			audio_path, effect = tmp_audio_path[1], None

		#torchaudio.initialize_sox()
		#sox = torchaudio.sox_effects.SoxEffectsChain()
		#if effect:
		#	sox.append_effect_to_chain(*effect)
		#sox.append_effect_to_chain('channels', 1)
		#sox.append_effect_to_chain('rate', sample_rate)
		#sox.set_input_file(audio_path)
		#signal, sample_rate_ = sox.sox_build_flow_effects()
		#signal = signal[0]
		#sox.clear_chain()
		#torchaudio.shutdown_sox()
		#print(['sox', '-V0', audio_path, '-b', '16', '-e', 'signed', '--endian', 'little', '-r', str(sample_rate), '-c', '1', '-t', 'raw', '-']  + ([effect[0], str(effect[1])] if effect else []))

		#signal, sample_rate_ = torch.as_tensor(bytearray(subprocess.check_output(['sox', '-V0', audio_path, '-b', '16', '-e', 'signed', '--endian', 'little', '-r', str(sample_rate), '-c', '1', '-t', 'raw', '-']  + ([effect[0], str(effect[1])] if effect else []))), dtype = torch.int16), sample_rate

		signal, sample_rate_ = torch.from_numpy(np.frombuffer(subprocess.check_output(['sox', '-V0', audio_path, '-b', '16', '-e', 'signed', '--endian', 'little', '-r', str(sample_rate), '-c', '1', '-t', 'raw', '-']  + ([effect[0], str(effect[1])] if effect else [])), dtype = np.int16)).to(torch.float32), sample_rate

		for audio_path in tmp_audio_path:
			os.remove(audio_path)
		if sample_rate is not None and sample_rate_ != sample_rate:
			signal, sample_rate_ = dataset.resample(signal, sample_rate_, sample_rate)
		if normalize:
			signal = models.normalize_signal(signal)
		if effect == []:
			signal, sample_rate = transform(signal, sample_rate) 
		return signal, sample_rate

class PitchShift(object):
	def __init__(self, n_steps = [-3, 3]):
		self.n_steps = n_steps

	def __call__(self, signal, sample_rate):
		return torch.from_numpy(librosa.effects.pitch_shift(signal.numpy(), sample_rate, fixed_or_uniform(self.n_steps))), sample_rate

class SpeedPerturbation(object):
	def __init__(self, rate = [0.8, 1.2]):
		self.rate = rate

	def __call__(self, signal, sample_rate):
		return torch.from_numpy(librosa.effects.time_stretch(signal.numpy(), fixed_or_uniform(self.rate))), sample_rate

class GainPerturbation(object):
	def __init__(self, gain_db = [-10, 10]):
		self.gain_db = gain_db

	def __call__(self, signal, sample_rate):
		return signal * (10. ** (fixed_or_uniform(self.gain_db) / 20.)), sample_rate

class AddWhiteNoise(object):
	def __init__(self, noise_level = 0.025):
		self.noise_level = noise_level

	def __call__(self, signal, sample_rate):
		noise = torch.randn_like(signal).clamp(-1, 1)
		noise_level = fixed_or_uniform(self.noise_level)
		return signal + noise * noise_level, sample_rate

class MixExternalNoise(object):
	def __init__(self, noise_level, noise_data_path, cache = False):
		self.noise_level = noise_level
		self.noise_data_path = noise_data_path
		self.noise_paths = list(map(str.strip, open(noise_data_path))) if noise_data_path is not None else []
		self.cache = {noise_path : dataset.read_wav(noise_path) for noise_path in self.noise_paths} if cache else {}

	def __call__(self, signal, sample_rate):
		noise_path = random.choice(self.noise_paths)
		noise_level = fixed_or_uniform(self.noise_level)
		noise, sample_rate_ = self.cache.get(noise_path) or dataset.read_wav(noise_path, sample_rate = sample_rate, max_duration = 1.0 + len(signal) / sample_rate)
		if sample_rate_ != sample_rate:
			noise, sample_rate_ = dataset.resample(noise, sample_rate_, sample_rate)
		noise = torch.cat([noise] * (1 + len(signal) // len(noise)))[:len(signal)]
		return signal + noise * noise_level, sample_rate

class Transcode(object):
	def __init__(self, codec = 'amr-nb', sample_rate = 8000, tmpdir = '/dev/shm'):
		self.sample_rate = int(sample_rate)
		self.codec = codec
		self.tmpdir = tmpdir
	
	def __call__(self, audio_path, sample_rate, normalize = True):
		tmp_audio_path = [tempfile.mkstemp(suffix = '.' + self.codec, dir = self.tmpdir)[1], tempfile.mkstemp(suffix = '.wav', dir = self.tmpdir)[1]]
		subprocess.check_call(['sox', '-V0', audio_path, '-t', self.codec, '-r', str(self.sample_rate), tmp_audio_path[0]])
		subprocess.check_call(['sox', '-V0', tmp_audio_path[0], '-t', 'wav', tmp_audio_path[1]])
		signal, sample_rate_ = torchaudio.load(tmp_audio_path)

		for audio_path in tmp_audio_path:
			os.remove(audio_path)
		if normalize:
			signal = models.normalize_signal(signal)
		if sample_rate_ != sample_rate:
			sample_rate_, signal = dataset.resample(signal, sample_rate_, sample_rate)
		return signal, sample_rate

class SpecLowPass(object):
	def __init__(self, freq):
		self.freq = int(freq)

	def __call__(self, spect, sample_rate):
		mel_cut, mel_max = librosa.hz_to_mel(self.freq), librosa.hz_to_mel(sample_rate / 2)
		n_freq = int(len(spect) * mel_cut / mel_max)
		spect[n_freq:] = 0
		return spect, sample_rate

class SpecHighPass(object):
	def __init__(self, freq):
		self.freq = int(freq)

	def __call__(self, spect, sample_rate):
		mel_cut, mel_max = librosa.hz_to_mel(self.freq), librosa.hz_to_mel(sample_rate / 2),  
		n_freq = int(len(spect) * mel_cut / mel_max)
		spect[:n_freq] = 0
		return spect, sample_rate

class SpecAugment(object):
	def __init__(self, n_freq_mask = 2, n_time_mask = 2, width_freq_mask = 6, width_time_mask = 6, replace_strategy = None):
		# fb code: https://github.com/facebookresearch/wav2letter/commit/04c3d80bf66fe749466cd427afbcc936fbdec5cd
		# width_freq_mask = 27, width_time_mask = 100, and n_freq_mask/n_time_mask = 2
		# google code: https://github.com/tensorflow/lingvo/blob/master/lingvo/core/spectrum_augmenter.py#L37-L42
		# width_freq_mask = 10 and width_time_mask = 50, and n_freq_mask/n_time_mask = 2

		self.replace_strategy = replace_strategy
		self.n_time_mask = n_time_mask
		self.n_freq_mask = n_freq_mask
		self.width_time_mask = width_time_mask
		self.width_freq_mask = width_freq_mask

	def __call__(self, spect):
		replace_val = spect.mean() if self.replace_strategy == 'mean' else 0

		for idx in range(self.n_freq_mask):
			f = random.randint(0, self.width_freq_mask)
			f0 = random.randint(0, spect.shape[0] - f)
			spect[f0:f0 + f, :] = replace_val

		for idx in range(self.n_time_mask):
			t = random.randint(0, min(self.width_time_mask, spect.shape[1]))
			t0 = random.randint(0, spect.shape[1] - t)
			spect[:, t0:t0 + t] = replace_val

		return spect

SOX = lambda prob = 1.0: RandomComposeSox([], prob)
AWN = lambda prob = 1.0: RandomComposeSox([AddWhiteNoise()], prob)
PS = lambda prob = 1.0: RandomComposeSox([PitchShift()], prob)
SP = lambda prob = 1.0: RandomComposeSox([SpeedPerturbation()], prob)
AMRNB = lambda prob = 1.0: RandomComposeSox([Transcode('amr-nb')], prob)
GSM = lambda prob = 1.0: RandomComposeSox([Transcode('gsm')], prob)
PSSPAMRNB = lambda prob = 1.0: RandomComposeSox([PitchShift(), SpeedPerturbation(), Transcode('amr-nb')], prob)
