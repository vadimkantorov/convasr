import random
import torch
import dataset
import librosa
import torchaudio
import models

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
		return '_'.join(t.__class__.__name__ + ('__'.join(str(y) for x in t.__dict__.values() for y in (x if isinstance(x, list) else [x]))) for t in self.transforms)

class RandomComposeSox(RandomCompose):
	def __init__(self, transforms, prob):
		super().__init__(transforms, prob)
		self.initialize_sox = True

	def __call__(self, audio_path, sample_rate, normalize = True):
		if self.initialize_sox:
			torchaudio.initialize_sox()
			self.initialize_sox = False
		sox = torchaudio.sox_effects.SoxEffectsChain()
		sox.set_input_file(audio_path)
		effect = None
		if random.random() < self.prob:
			transform = random.choice(self.transforms)
			effect = ['pitch', fixed_or_uniform(transform.n_steps) * 100] if isinstance(transform, PitchShift) else ['tempo', fixed_or_uniform(transform.rate)] if isinstance(transform, SpeedPerturbation) else ['gain', fixed_or_uniform(transform.gain_db)] if isinstance(transform, GainPerturbation) else []
			if effect:
				sox.append_effect_to_chain(*effect)
		sox.append_effect_to_chain('rate', sample_rate)
		sox.append_effect_to_chain('channels', 1)
		signal = sox.sox_build_flow_effects()[0][0]
		if normalize:
			signal = models.normalize_signal(signal)
		if effect == []:
			signal, sample_rate = transform(signal, sample_rate) 
		return signal, sample_rate

	def __str__(self):
		return 'SOXx' + RandomCompose.__str__(self)

class PitchShift(object):
	def __init__(self, n_steps = [-3, 4]):
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
	def __init__(self, noise_level, noise_data_path):
		self.noise_level = noise_level
		self.noise_data_path = noise_data_path
		self.noise_paths = list(map(str.strip, open(noise_data_path))) if noise_data_path is not None else []

	def __call__(self, signal, sample_rate):
		noise_path = random.choice(self.noise_paths)
		noise_level = fixed_or_uniform(self.noise_level)
		noise, sample_rate = dataset.read_wav(noise_path, sample_rate = sample_rate, max_duration = 1.0 + len(signal) / sample_rate)
		noise = torch.cat([noise] * (1 + len(signal) // len(noise)))[:len(signal)]
		return signal + noise * noise_level, sample_rate

class SpecLowPass(object):
	def __init__(self, freq, sample_rate):
		self.freq = freq
		self.sample_rate = sample_rate

	def __call__(self, spect):
		n_low_freq = int(len(spect) * freq / (sample_rate / 2))
		spect[n_low_freq:] = 0
		return spect

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

class SpecCutOut(object):
	def __init__(self, cutout_rect_freq = 25, cutout_rect_time = 60, cutout_rect_regions = 0):
		self.cutout_rect_regions = cutout_rect_regions
		self.cutout_rect_time = cutout_rect_time
		self.cutout_rect_freq = cutout_rect_freq

	def __call__(self, spect):
		for i in range(self.cutout_rect_regions):
			cutout_rect_x = random.randint(0, spect.shape[-2] - self.cutout_rect_freq)
			cutout_rect_y = random.randint(0, spect.shape[-1] - self.cutout_rect_time)
			spect[cutout_rect_x:cutout_rect_x + self.cutout_rect_freq, cutout_rect_y:cutout_rect_y + self.cutout_rect_time] = 0
		return spect

def fixed_or_uniform(r):
	return random.uniform(*r) if isinstance(r, list) else r

AWNSPGPPS = lambda prob = 0.3: RandomCompose([AddWhiteNoise(), SpeedPerturbation(), GainPerturbation(), PitchShift()], prob)

SOXAWNSPGPPS = lambda prob = 0.3: RandomComposeSox([AddWhiteNoise(), SpeedPerturbation(), GainPerturbation(), PitchShift()], prob)

SOXAWN = lambda prob = 1.0: RandomComposeSox([AddWhiteNoise()], prob)
#SOXPS = lambda prob = 1.0: RandomComposeSox([PitchShift(-3)], prob)
#SOXSP = lambda prob = 1.0: RandomComposeSox([SpeedPerturbation(0.8)], prob)
#SOXGP = lambda prob = 1.0: RandomComposeSox([GainPerturbation(-50)], prob)
