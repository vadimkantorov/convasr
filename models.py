import os
import math
import collections
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

class ConvSamePadding(nn.Sequential):
	def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, bias, groups, separable):
		padding = dilation * max(1, kernel_size // 2)
		if separable:
			assert dilation == 1
			super().__init__(
				nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dgroups = in_channels, bias = bias),
				nn.Conv1d(out_channels, out_channels, kernel_size = 1, bias = bias)
			)
		else:
			super().__init__(
				nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias)
			)

class ConvBN(nn.Module):
	def __init__(self, num_channels, kernel_size, stride = 1, dropout = 0, batch_norm_momentum = 0.1, groups = 1, num_channels_residual = [], repeat = 1, dilation = 1, separable = False, temporal_mask = True, inplace = False, nonlinearity = 'relu'):
		super().__init__()
		self.conv = nn.ModuleList(ConvSamePadding(num_channels[0] if i == 0 else num_channels[1], num_channels[1], kernel_size = kernel_size, stride = stride, dilation = dilation, separable = separable, bias = False, groups = groups) for i in range(repeat))
		self.bn = nn.ModuleList(ActivatedBatchNorm(num_channels[1], momentum = batch_norm_momentum, nonlinearity = nonlinearity, inplace = inplace, dropout = dropout) for i in range(repeat))
		self.conv_residual = nn.ModuleList(nn.Conv1d(in_channels, num_channels[1], kernel_size = 1) for in_channels in num_channels_residual)
		self.bn_residual = nn.ModuleList(ActivatedBatchNorm(num_channels[1], momentum = batch_norm_momentum, nonlinearity = None, inplace = inplace) for in_channels in num_channels_residual)
		self.temporal_mask = temporal_mask

	def forward(self, x, lengths_fraction = None, residual = []):
		y = x
		for i, (conv, bn) in enumerate(zip(self.conv, self.bn)):
			y = bn(conv(y), residual = [bn(conv(r)) for conv, bn, r in zip(self.conv_residual, self.bn_residual, residual)] if i == len(self.conv) - 1 else [])
			y = y * temporal_mask(y, lengths_fraction = lengths_fraction) if (self.temporal_mask and lengths_fraction is not None) else y
		return y

class JasperNet(nn.ModuleList):
	def __init__(self, num_classes, num_input_features, repeat = 3, num_subblocks = 1, dilation = 1, dropout = 'ignored', dropout_small = 0.2, dropout_medium = 0.3, dropout_large = 0.4, separable = False):
		dropout_small = dropout_small if dropout != 0 else 0
		dropout_medium = dropout_medium if dropout != 0 else 0
		dropout_large = dropout_large if dropout != 0 else 0

		prologue = [ConvBN(kernel_size = 11, num_channels = (num_input_features, 256), dropout = dropout_small, stride = 2)]
		
		if num_subblocks == 1:
			backbone = [
				ConvBN(kernel_size = 11, num_channels = (256, 256), dropout = dropout_small,  repeat = repeat, separable = separable, num_channels_residual = [256]),
				ConvBN(kernel_size = 13, num_channels = (256, 384), dropout = dropout_small,  repeat = repeat, separable = separable, num_channels_residual = [256, 256]),
				ConvBN(kernel_size = 17, num_channels = (384, 512), dropout = dropout_small,  repeat = repeat, separable = separable, num_channels_residual = [256, 256, 384]),
				ConvBN(kernel_size = 21, num_channels = (512, 640), dropout = dropout_medium, repeat = repeat, separable = separable, num_channels_residual = [256, 256, 384, 512]),
				ConvBN(kernel_size = 25, num_channels = (640, 768), dropout = dropout_medium, repeat = repeat, separable = separable, num_channels_residual = [256, 256, 384, 512, 640])
			]
		elif num_subblocks == 2:
			backbone = [
				ConvBN(kernel_size = 11, num_channels = (256, 256), dropout = dropout_small, repeat = repeat, separable = separable, num_channels_residual = [256]),
				ConvBN(kernel_size = 11, num_channels = (256, 256), dropout = dropout_small, repeat = repeat, separable = separable, num_channels_residual = [256, 256]),
				
				ConvBN(kernel_size = 13, num_channels = (256, 384), dropout = dropout_small, repeat = repeat, separable = separable, num_channels_residual = [256, 256, 256]),
				ConvBN(kernel_size = 13, num_channels = (384, 384), dropout = dropout_small, repeat = repeat, separable = separable, num_channels_residual = [256, 256, 256, 384]),
				
				ConvBN(kernel_size = 17, num_channels = (384, 512), dropout = dropout_small, repeat = repeat, separable = separable, num_channels_residual = [256, 256, 256, 384, 384]),
				ConvBN(kernel_size = 17, num_channels = (512, 512), dropout = dropout_small, repeat = repeat, separable = separable, num_channels_residual = [256, 256, 256, 384, 384, 512]),
				
				ConvBN(kernel_size = 21, num_channels = (512, 640), dropout = dropout_medium, repeat = repeat, separable = separable, num_channels_residual = [256, 256, 256, 384, 384, 512, 512]),
				ConvBN(kernel_size = 21, num_channels = (640, 640), dropout = dropout_medium, repeat = repeat, separable = separable, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640]),
				
				ConvBN(kernel_size = 25, num_channels = (640, 768), dropout = dropout_medium, repeat = repeat, separable = separable, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640, 640]),
				ConvBN(kernel_size = 25, num_channels = (768, 768), dropout = dropout_medium, repeat = repeat, separable = separable, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640, 640, 768])
			]

		epilogue = [
			ConvBN(kernel_size = 29, num_channels = (768, 896), dropout = dropout_large, dilation = dilation),
			ConvBN(kernel_size = 1, num_channels = (896, 1024), dropout = dropout_large),
			nn.Conv1d(1024, num_classes, kernel_size = 1)
		]
		super().__init__(prologue + backbone + epilogue)

	def forward(self, x, lengths_fraction):
		residual = []
		for i, subblock in enumerate(list(self)[:-1]):
			x = subblock(x, residual = residual if i < len(self) - 3 else [], lengths_fraction = lengths_fraction)
			residual.append(x)
		logits = self[-1](x)
		return logits, compute_output_lengths(logits, lengths_fraction)

class Wav2Letter(nn.Sequential):
	# TODO: use hardtanh 20
	def __init__(self, num_classes, num_input_features, dilation = 2):
		super().__init__(
			ConvBN(kernel_size = 11, num_channels = (num_input_features, 256), stride = 2, padding = 5),
			ConvBN(kernel_size = 11, num_channels = (256, 256), repeat = 3, padding = 5),
			ConvBN(kernel_size = 13, num_channels = (256, 384), repeat = 3, padding = 6),
			ConvBN(kernel_size = 17, num_channels = (384, 512), repeat = 3, padding = 8),
			ConvBN(kernel_size = 21, num_channels = (512, 640), repeat = 3, padding = 10),
			ConvBN(kernel_size = 25, num_channels = (640, 768), repeat = 3, padding = 12),
			ConvBN(kernel_size = 29, num_channels = (768, 896), repeat = 1, padding = 28, dilation = dilation),
			ConvBN(kernel_size = 1, num_channels = (896, 1024), repeat = 1),
			nn.Conv1d(1024, num_classes, kernel_size = 1)
		)

	def forward(self, x, lengths_fraction):
		logits = super().forward(x)
		return logits, compute_output_lengths(logits, lengths_fraction)

class Wav2LetterRu(nn.Sequential):
	def __init__(self, num_classes, num_input_features, dropout = 0.2, width_large = 2048, kernel_size_large = 29):
		super().__init__(
			ConvBN(kernel_size = 13, num_channels = (num_input_features, 768), stride = 2, dropout = dropout),
			
			ConvBN(kernel_size = 13, num_channels = (768, 768), stride = 1, dropout = dropout),
			ConvBN(kernel_size = 13, num_channels = (768, 768), stride = 1, dropout = dropout),
			ConvBN(kernel_size = 13, num_channels = (768, 768), stride = 1, dropout = dropout),
			ConvBN(kernel_size = 13, num_channels = (768, 768), stride = 1, dropout = dropout),
			ConvBN(kernel_size = 13, num_channels = (768, 768), stride = 1, dropout = dropout),
			ConvBN(kernel_size = 13, num_channels = (768, 768), stride = 1, dropout = dropout),

			ConvBN(kernel_size = kernel_size_large, num_channels = (768, width_large), stride = 1, dropout = dropout),
			ConvBN(kernel_size = 1,  num_channels = (width_large, width_large),stride = 1, dropout = dropout),
			nn.Conv1d(width_large, num_classes, kernel_size = 1, stride = 1)
		)

	def forward(self, x, lengths_fraction):
		logits = super().forward(x)
		return logits, compute_output_lengths(logits, lengths_fraction)

class BabbleNet(nn.Sequential):
	def __init__(self, num_classes, num_input_features, dropout = 0.2, repeat = 1, batch_norm_momentum = 0.1):
		super().__init__(
			ConvBN(kernel_size = 13, num_channels = (num_input_features, 768), stride = 2, dropout = dropout),

			ConvBN(kernel_size = 13, num_channels = (768, 192), stride = 1, dropout = dropout),
			InvertedResidual(kernel_size = 13, num_channels = (192, 192), stride = 1, dropout = dropout, expansion = 4),
			InvertedResidual(kernel_size = 13, num_channels = (192, 192), stride = 1, dropout = dropout, expansion = 4),
			InvertedResidual(kernel_size = 13, num_channels = (192, 192), stride = 1, dropout = dropout, expansion = 4),
			InvertedResidual(kernel_size = 13, num_channels = (192, 192), stride = 1, dropout = dropout, expansion = 4),
			InvertedResidual(kernel_size = 13, num_channels = (192, 192), stride = 1, dropout = dropout, expansion = 4),
			ConvBN(kernel_size = 13, num_channels = (192, 768), stride = 1, dropout = dropout),

			ConvBN(kernel_size = 31, num_channels = (768, 2048), stride = 1, dropout = dropout),
			ConvBN(kernel_size = 1,  num_channels = (2048, 2048), stride = 1, dropout = dropout),
			nn.Conv1d(2048, num_classes, kernel_size = 1)
		)

	def forward(self, x, lengths_fraction):
		logits = super().forward(x)
		return logits, compute_output_lengths(logits, lengths_fraction)

class ActivatedBatchNorm(nn.modules.batchnorm._BatchNorm):
	def __init__(self, *args, nonlinearity = None, inplace = False, dropout = 0, squeeze_and_excite = None, **kwargs):
		super().__init__(*args, **kwargs)
		self.nonlinearity = nonlinearity
		self.inplace = inplace
		self.dropout = dropout
		self.squeeze_and_excite = squeeze_and_excite

	def _check_input_dim(self, input):
		return True

	def forward(self, input, residual = []):
		assert not (self.inplace and self.nonlinearity == 'relu')

		if self.inplace:
			y = ActivatedBatchNorm.Function.apply(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.momentum, self.training, self.nonlinearity, *residual)
			y = F.dropout(y, p = self.dropout, training = self.training)
		else:
			y = super().forward(input)
			s = y * (self.squeeze_and_excite(y) if self.squeeze_and_excite is not None else 1)
			y = y + sum(residual) #(functools.reduce(lambda acc, x: acc.add_(x), residual, torch.zeros_like(residual[0])) if len(residual) > 1 else sum(residual))
			if self.nonlinearity == 'relu':
				y = relu_dropout(y, p = self.dropout, inplace = True, training = self.training)
			elif self.nonlinearity and self.nonlinearity[0] == 'leaky_relu':
				y = F.leaky_relu_(y, self.nonlinearity[1])
				y = F.dropout(y, p = self.dropout, training = self.training)
		return y

	class Function(torch.autograd.function.Function):
		@staticmethod
		def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, training, nonlinearity, *residual):
			self.nonlinearity = nonlinearity
			assert input.is_contiguous()
			
			mean, var = torch.batch_norm_update_stats(input, running_mean, running_var, momentum) if training else (running_mean, running_var) 
			invstd = (var + eps).sqrt_().reciprocal_()

			output = torch.batch_norm_elemt(input, input, weight, bias, mean, invstd, 0)
			for r in residual:
				output += r

			if self.nonlinearity and self.nonlinearity[0] == 'leaky_relu':
				F.leaky_relu_(output, self.nonlinearity[1])

			self.save_for_backward(output, weight, bias, mean, invstd, *residual)
			return output

		@staticmethod
		def backward(self, grad_output):
			saved_output, weight, bias, mean, invstd, *residual = self.saved_tensors
			assert grad_output.is_contiguous() and saved_output.is_contiguous()

			if self.nonlinearity and self.nonlinearity[0] == 'leaky_relu':
				mask = torch.ones_like(grad_output).masked_fill_(saved_output < 0, self.nonlinearity[1])
				grad_output *= mask
				saved_output /= mask

			for r in residual:
				saved_output -= r

			saved_input = torch.batch_norm_elemt(saved_output, saved_output, 1. / invstd, mean, bias, 1. / weight, 0)
			mean_dy, mean_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(grad_output, saved_input, mean, invstd,	weight,	self.needs_input_grad[0], self.needs_input_grad[1],	self.needs_input_grad[2])
			grad_input = torch.batch_norm_backward_elemt(grad_output, saved_input, mean, invstd, weight, mean_dy, mean_dy_xmu)

			return (grad_input, grad_weight, grad_bias, None, None, None, None, None, None) + tuple([grad_output] * len(residual))

class SqueezeAndExcite(nn.Sequential):
	def __init__(self, out_channels, ratio = 0.25):
		se_channels = int(out_channels * ratio)
		super().__init__(
			nn.AdaptiveAvgPool1d(1),
			nn.Conv1d(out_channels, se_channels, kernel_size = 1),
			nn.ReLU(inplace = True),
			nn.Conv1d(se_channels, out_channels, kernel_size = 1),
			nn.Sigmoid()
		)

def relu_dropout(x, p = 0, inplace = False, training = False):
	if not training or p == 0:
		return x.clamp_(min = 0) if inplace else x.clamp(min = 0)
	
	p1m = 1 - p
	mask = torch.rand_like(x) < p1m
	mask &= (x > 0)
	mask.logical_not_()
	return x.masked_fill_(mask, 0).div_(p1m) if inplace else (x.masked_fill(mask, 0) / p1m)

def logfbank(signal, sample_rate, window_size, window_stride, window, num_input_features, dither = 1e-5, preemph = 0.97, normalize = True, eps = 1e-20):
	signal = normalize_signal(signal)
	signal = torch.cat([signal[..., :1], signal[..., 1:] - preemph * signal[..., :-1]], dim = -1)
	win_length, hop_length = int(window_size * sample_rate), int(window_stride * sample_rate)
	n_fft = 2 ** math.ceil(math.log2(win_length))
	signal += dither * torch.randn_like(signal)
	window = getattr(torch, window)(win_length, periodic = False).type_as(signal)
	#mel_basis = torchaudio.functional.create_fb_matrix(n_fft, n_mels = num_input_features, fmin = 0, fmax = int(sample_rate/2)).t() # when https://github.com/pytorch/audio/issues/287 is fixed
	mel_basis = torch.from_numpy(librosa.filters.mel(sample_rate, n_fft, n_mels=num_input_features, fmin=0, fmax=int(sample_rate/2))).type_as(signal)
	power_spectrum = torch.stft(signal, n_fft, hop_length = hop_length, win_length = win_length, window = window, pad_mode = 'reflect', center = True).pow(2).sum(dim = -1)
	features = torch.log(torch.matmul(mel_basis, power_spectrum) + eps)
	return normalize_features(features) if normalize else features 

def temporal_mask(x, lengths = None, lengths_fraction = None):
	lengths = lengths if lengths is not None else compute_output_lengths(x, lengths_fraction)
	return (torch.arange(x.shape[-1], device = x.device, dtype = lengths.dtype).unsqueeze(0) < lengths.unsqueeze(1)).view(x.shape[:1] + (1, )*(len(x.shape) - 2) + x.shape[-1:])

def entropy(log_probs, lengths = None, dim = 1, eps = 1e-9, sum = True, keepdim = False):
	e = -(log_probs.exp() * log_probs).sum(dim = dim, keepdim = keepdim)
	if lengths is not None:
		e = e * temporal_mask(e, lengths)
	return (e.sum(dim = -1) / (eps + lengths.type_as(log_probs)) if lengths is not None else e.mean(dim = -1)) if sum else e

def margin(log_probs, dim = 1):
	probs = log_probs.exp()
	return torch.sub(*probs.topk(2, dim = dim).values)

def compute_output_lengths(x, lengths_fraction):
	return (lengths_fraction * x.shape[-1]).ceil().int()

def compute_capacity(model):
	return sum(map(torch.numel, model.parameters()))

def normalize_signal(signal, dim = -1, eps = 1e-5):
	signal = signal.to(torch.float32)
	return signal / (signal.abs().max(dim = dim, keepdim = True).values + eps)

def normalize_features(features, dim = -1, eps = 1e-20):
	return (features - features.mean(dim = dim, keepdim = True)) / (features.std(dim = dim, keepdim = True) + eps)
