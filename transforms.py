import os
import math
import tempfile
import random
import subprocess
import torch
import librosa
import torchaudio
import models
import numpy as np

fixed_or_uniform = lambda r: random.uniform(*r) if isinstance(r, list) else r
fixed_or_choice = lambda r: random.choice(r) if isinstance(r, list) else r

class RandomCompose:
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

class SoxAug(RandomCompose):
	def __init__(self, transforms, prob, bug = None):
		super().__init__(transforms, prob)
		self.bug = bug

	def __call__(self, audio_path, sample_rate, normalize = True, defaults = dict(pitch = [-300, 300], tempo = [0.8, 1.2], gain = [-10, 10]), tmpdir = '/dev/shm'):
		effect = None
		tuple_if_str = lambda t: (t, defaults.get(t)) if isinstance(t, str) else t
		if self.transforms and random.random() < self.prob:
			transform = tuple_if_str(random.choice(self.transforms))
			effect = ([transform[0], fixed_or_choice(transform[1])] if transform[0] in defaults else transform[0]) if isinstance(transform, tuple) else []

		tmp_audio_path = []
		if effect and isinstance(effect, str) and effect.startswith('transcode'):
			codec = effect.split('_')[1]
			tmp_audio_path = [tempfile.mkstemp(suffix = '.' + codec, dir = tmpdir)[1], tempfile.mkstemp(suffix = '.wav', dir = tmpdir)[1]]
			subprocess.check_call(['sox', '-V0', audio_path, '-t', codec, '-r', str(sample_rate), tmp_audio_path[0]])
			if self.bug == 'SoxEffectsChain':
				subprocess.check_call(['sox', '-V0', tmp_audio_path[0], '-t', 'wav', tmp_audio_path[1]])
				audio_path = tmp_audio_path[1]
			else:
				audio_path = tmp_audio_path[0]
			effect = None

		if self.bug == 'SoxEffectsChain':
			torchaudio.initialize_sox()
			sox = torchaudio.sox_effects.SoxEffectsChain()
			if effect:
				sox.append_effect_to_chain(*effect)
			sox.append_effect_to_chain('channels', 1)
			sox.append_effect_to_chain('rate', sample_rate)
			sox.set_input_file(audio_path)
			signal, sample_rate_ = sox.sox_build_flow_effects()
			signal = signal[0]
			sox.clear_chain()
			torchaudio.shutdown_sox()

		elif self.bug == 'as_tensor':
			signal, sample_rate_ = torch.as_tensor(bytearray(subprocess.check_output(['sox', '-V0', audio_path, '-b', '16', '-e', 'signed', '--endian', 'little', '-r', str(sample_rate), '-c', '1', '-t', 'raw', '-']  + ([effect[0], str(effect[1])] if effect else []))), dtype = torch.int16), sample_rate

		else:
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

class Quantization:
	def __init__(self, quantization_channels = 16):
		self.quantization_channels = quantization_channels
	
	def __call__(self, signal, sample_rate):
		quantization_channels = fixed_or_choice(self.quantization_channels)
		quantized = torchaudio.functional.mu_law_encoding(signal, quantization_channels)
		return torchaudio.functional.mu_law_decoding(quantized, quantization_channels), sample_rate

class AddWhiteNoise:
	def __init__(self, noise_level = 0.025):
		self.noise_level = float(noise_level)

	def __call__(self, signal, sample_rate, dataset_name = ''):
		if 'tts' not in dataset_name:
			return signal, sample_rate

		noise = torch.randn_like(signal).clamp(-1, 1)
		noise_level = fixed_or_uniform(self.noise_level)
		return signal + noise * noise_level, sample_rate

class MixExternalNoise:
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

class SpecLowPass:
	def __init__(self, freq):
		self.freq = int(freq)

	def __call__(self, spect, sample_rate):
		mel_cut, mel_max = librosa.hz_to_mel(self.freq), librosa.hz_to_mel(sample_rate / 2)
		n_freq = int(len(spect) * mel_cut / mel_max)
		spect[n_freq:] = 0
		return spect, sample_rate

class SpecHighPass:
	def __init__(self, freq):
		self.freq = int(freq)

	def __call__(self, spect, sample_rate):
		mel_cut, mel_max = librosa.hz_to_mel(self.freq), librosa.hz_to_mel(sample_rate / 2),  
		n_freq = int(len(spect) * mel_cut / mel_max)
		spect[:n_freq] = 0
		return spect, sample_rate

class SpecAugment:
	def __init__(self, n_freq_mask = 2, n_time_mask = 5, width_freq_mask = 6, width_time_mask = 10):
		# fb code: https://github.com/facebookresearch/wav2letter/commit/04c3d80bf66fe749466cd427afbcc936fbdec5cd
		# width_freq_mask = 27, width_time_mask = 100, and n_freq_mask/n_time_mask = 2
		# google code: https://github.com/tensorflow/lingvo/blob/master/lingvo/core/spectrum_augmenter.py#L37-L42
		# width_freq_mask = 10 and width_time_mask = 50, and n_freq_mask/n_time_mask = 2

		self.n_time_mask = n_time_mask
		self.n_freq_mask = n_freq_mask
		self.width_time_mask = width_time_mask
		self.width_freq_mask = width_freq_mask

	def __call__(self, spect, sample_rate, replace_val = 0):
		for idx in range(self.n_freq_mask):
			f = random.randint(0, self.width_freq_mask)
			f0 = random.randint(0, spect.shape[0] - f)
			spect[f0:f0 + f, :] = replace_val

		for idx in range(self.n_time_mask):
			t = random.randint(0, min(self.width_time_mask, spect.shape[1]))
			t0 = random.randint(0, spect.shape[1] - t)
			spect[:, t0:t0 + t] = replace_val

		return spect, sample_rate

class SpecPerlinNoise:
	def __init__(self, noise_level = 0.4):
		self.noise_level = float(noise_level)

	def __call__(self, spect, sample_rate, kernel_size = 8, octaves=4, persistence=0.5, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
		def rand_perlin_2d(spect, kernel_size):
			shape = [(1 + s // (2 * kernel_size)) * (2 * kernel_size) for s in spect.shape]

			delta = (kernel_size / shape[0], kernel_size / shape[1])
			d = (shape[0] // kernel_size, shape[1] // kernel_size)
			grid = torch.stack(torch.meshgrid(torch.arange(0, kernel_size, delta[0]), torch.arange(0, kernel_size, delta[1])), dim = -1) % 1

			angles = 2*math.pi*torch.rand(kernel_size+1, kernel_size+1)
			gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
			g00 = gradients[0:-1,0:-1].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
			g10 = gradients[1:  ,0:-1].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
			g01 = gradients[0:-1,1:  ].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
			g11 = gradients[1:  ,1:  ].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)

			sum_stack = lambda grid, grad, d0, d1: (torch.stack((grid[:spect.shape[0],:spect.shape[1],0] + d0  , grid[:spect.shape[0],:spect.shape[1],1] + d1  ), dim = -1) * grad[:spect.shape[0], :spect.shape[1]]).sum(dim = -1)
			n00 = sum_stack(grid, g00, 0, 0)
			n10 = sum_stack(grid, g10, -1, 0)
			n01 = sum_stack(grid, g01, 0, -1)
			n11 = sum_stack(grid, g11, -1,-1)
			t = fade(grid[:spect.shape[0], :spect.shape[1]])
			n0 = torch.lerp(n00, n10, t[:, :, 0])
			n1 = torch.lerp(n01, n11, t[:, :, 0])
			return math.sqrt(2) * torch.lerp(n0, n1, t[:, :, 1])

		noise = torch.zeros_like(spect)
		frequency = 1
		amplitude = 1
		for _ in range(octaves):
			noise += amplitude * rand_perlin_2d(spect, frequency*kernel_size)
			frequency *= 2
			amplitude *= persistence
		return spect + self.noise_level * noise, sample_rate

AWN = lambda prob = 1.0: SoxAug([AddWhiteNoise()], prob)
PS = lambda prob = 1.0: SoxAug(['pitch'], prob)
SP = lambda prob = 1.0: SoxAug(['tempo'], prob)
AMRNB = lambda prob = 1.0: SoxAug(['transcode_amr-nb'], prob)
GSM = lambda prob = 1.0: SoxAug(['transcode_gsm'], prob)
PSSPAMRNB = lambda prob = 1.0: SoxAug(['pitch', 'tempo', 'transcode_amr-nb'], prob)

PS_BUG_SoxEffectsChain = lambda prob = 1.0: SoxAug(['pitch'], prob, bug = 'SoxEffectsChain')
PS_BUG_as_tensor = lambda prob = 1.0: SoxAug(['pitch'], prob, bug = 'as_tensor')
