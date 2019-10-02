import os
import math
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

class BatchNormInplaceFunction(torch.autograd.function.Function):
	@staticmethod
	def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, training, activation):
		self.activation = activation

		input = input.contiguous()
		if training:
			#mean, invstd = torch.batch_norm_stats(input, eps)
			mean, var = torch.batch_norm_update_stats(input, running_mean, running_var, momentum)
			invstd = var.add_(eps).sqrt_().reciprocal_()
		else:
			mean, var = running_mean, running_var
			invstd = var.add(eps).sqrt().reciprocal()

		out = torch.batch_norm_elemt(input, input, weight, bias, mean, invstd, eps)
		if self.activation and self.activation[0] == 'leaky_relu':
			out = F.leaky_relu_(out, *activation[1:])

		if training:
			self.save_for_backward(out, weight, bias, mean, invstd)
		return out

	@staticmethod
	def backward(self, grad_output):
		saved_input, weight, bias, mean, invstd = self.saved_tensors
		grad_output = grad_output.contiguous()
		saved_input = saved_input.contiguous()
		reshape = lambda x: x.view(-1, *[1]*(len(saved_input.shape) - 2))

		if self.activation and self.activation[0] == 'leaky_relu':
			mask = torch.ones_like(grad_output).masked_fill_(saved_input < 0, self.activation[1])
			grad_output *= mask
			saved_input /= mask

		saved_input -= reshape(bias)
		saved_input /= reshape(weight)
		saved_input /= reshape(invstd)
		saved_input += reshape(mean)
		
		mean_dy, mean_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
			grad_output,
			saved_input,
			mean,
			invstd,
			weight,
			self.needs_input_grad[0],
			self.needs_input_grad[1],
			self.needs_input_grad[2]
		)

		grad_input = torch.batch_norm_backward_elemt(
			grad_output,
			saved_input,
			mean,
			invstd,
			weight,
			mean_dy,
			mean_dy_xmu
		)

		return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

class BatchNormInplace(nn.modules.batchnorm._BatchNorm):
	def __init__(self, *args, activation = ('leaky_relu', 0.01), **kwargs):
		super().__init__(*args, **kwargs)
		self.activation = activation

	def forward(self, input):
		return BatchNormInplaceFunction.apply(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.momentum, self.training, self.activation)

#TODO: apply conv masking
class ConvBN(nn.ModuleDict):
	def __init__(self, num_channels, kernel_size, stride = 1, dropout = 0, batch_norm_momentum = 0.1, residual = False, groups = 1, num_channels_residual = [], repeat = 1, dilation = 1, separable = False):
		super().__init__(dict(
			conv = nn.ModuleList(nn.Conv1d(num_channels[0] if i == 0 else num_channels[1], num_channels[1], kernel_size = kernel_size, stride = stride, padding = dilation * max(1, kernel_size // 2), bias = False, groups = groups, dilation = dilation) for i in range(repeat)),
			bn = nn.ModuleList(BatchNormInplace(num_channels[1], momentum = batch_norm_momentum, activation = ('leaky_relu', 0.01) if (not residual) or repeat == 1 or i != repeat - 1 else None) for i in range(repeat)),
			conv_residual = nn.ModuleList(nn.Conv1d(in_channels, num_channels[1], kernel_size = 1) for in_channels in num_channels_residual),
			bn_residual = nn.ModuleList(BatchNormInplace(num_channels[1], momentum = batch_norm_momentum, activation = None) for in_channels in num_channels_residual)
		))
		self.residual = residual

	def forward(self, x, residual = []):
		y = x
		for conv, bn in zip(self.conv, self.bn):
			y = bn(conv(y))

		if residual:
			y = F.relu(y + sum(bn(conv(r)) for conv, bn, r in zip(self.conv_residual, self.bn_residual, residual)))
		
		elif self.residual and (self.conv[0].in_channels == self.conv[0].out_channels and y.shape[-1] == x.shape[-1]):
			y = F.relu(y + x)
		
		return y

class InvertedResidual(nn.Module):
	def __init__(self, kernel_size, num_channels, stride = 1, dilation = 1, dropout = 0.2, expansion = 1, squeeze_excitation_ratio = 0.25, batch_norm_momentum = 0.1, separable = True, simple = False, residual = True):
		super().__init__()
		in_channels, out_channels = num_channels
		exp_channels = in_channels * expansion if not simple else out_channels
		se_channels = int(exp_channels * squeeze_excitation_ratio)
		groups = exp_channels if (separable and not simple) else 1

		self.simple = simple
		self.expand = ConvBN(num_channels = (in_channels, exp_channels), stride = stride, batch_norm_momentum = batch_norm_momentum) if not simple else nn.Identity()
		self.conv = ConvBN(kernel_size = kernel_size, num_channels = (exp_channels if not simple else in_channels, exp_channels), stride = stride, groups = groups, dropout = dropout, batch_norm_momentum = batch_norm_momentum)
		self.squeeze_and_excite = nn.Sequential(
			nn.AdaptiveAvgPool1d(1),
			nn.Conv1d(exp_channels, se_channels, kernel_size = 1),
			nn.ReLU(inplace = True),
			nn.Conv1d(se_channels, exp_channels, kernel_size = 1),
			nn.Sigmoid()
		) if not simple else nn.Identity()
		self.reduce = ConvBN(num_channels = (exp_channels, out_channel), batch_norm_momentum = batch_norm_momentum, relu_dropout = False)
		self.residual = ConvBN(num_channels = (in_channels, out_channels), batch_norm_momentum = batch_norm_momentum, relu_dropout = False) if residual and (not simple or in_channels != out_channels) else nn.Identity() if residual else None
	
	def forward(self, x):
		if self.simple:
			return self.reduce(self.conv(x)) + self.residual(x) if self.residual is not None else self.conv(x)

		y = self.expand(x)
		y = self.conv(y)
		y = y * self.squeeze_and_excite(y)
		y = self.reduce(y)
		return y + self.residual(x)

class ReLUDropout(torch.nn.Dropout):
	def forward(self, input):
		if self.training and self.p > 0:
			p1m = 1. - self.p
			mask = torch.rand_like(input) < p1m
			mask *= (input > 0)
			return input.masked_fill_(~mask, 0).div_(p1m) if self.inplace else (input.masked_fill(~mask, 0) / p1m)
		else:
			return input.clamp_(min = 0) if self.inplace else input.clamp(min = 0)

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

	def forward(self, x, input_lengths_fraction):
		logits = super().forward(x)
		return logits, compute_output_lengths(logits, input_lengths_fraction)

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

	def forward(self, x, input_lengths_fraction):
		logits = super().forward(x)
		return logits, compute_output_lengths(logits, input_lengths_fraction)

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

	def forward(self, x, input_lengths_fraction):
		logits = super().forward(x)
		return logits, compute_output_lengths(logits, input_lengths_fraction)

class JasperNet(nn.ModuleList):
	def __init__(self, num_classes, num_input_features, repeat = 3, num_subblocks = 2, dilation = 1, dropout = 'ignored', dropout_small = 0.2, dropout_medium = 0.3, dropout_large = 0.4, separable = False):
		dropout_small = dropout_small if dropout != 0 else 0
		dropout_medium = dropout_medium if dropout != 0 else 0
		#dropout_large = dropout_large if dropout != 0 else 0

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

	def forward(self, x, input_lengths_fraction):
		residual = []
		for i, subblock in enumerate(list(self)[:-1]):
			x = subblock(x, residual = residual if i < len(self) - 3 else [])
			residual.append(x)
		logits = self[-1](x)
		return logits, compute_output_lengths(logits, input_lengths_fraction)

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

def normalize_signal(signal, dim = -1, eps = 1e-5):
	signal = signal.to(torch.float32)
	return signal / (signal.abs().max(dim = dim, keepdim = True).values + eps)

def normalize_features(features, dim = -1, eps = 1e-20):
	return (features - features.mean(dim = dim, keepdim = True)) / (features.std(dim = dim, keepdim = True) + eps)

def temporal_mask(x, lengths = None, lengths_fraction = None):
	lengths = lengths if lengths is not None else compute_output_lengths(x, lenghts_fraction)
	mask = torch.ones_like(x)
	for m, l in zip(mask, lengths):
		m[..., l:].zero_()
	return mask

def entropy(log_probs, lengths = None, dim = 1, eps = 1e-9, sum = True, keepdim = False):
	e = -(log_probs.exp() * log_probs).sum(dim = dim, keepdim = keepdim)
	if lengths is not None:
		e = e * temporal_mask(e, lengths)
	return (e.sum(dim = -1) / (eps + lengths.type_as(log_probs)) if lengths is not None else e.mean(dim = -1)) if sum else e

def margin(log_probs, dim = 1):
	probs = log_probs.exp()
	return torch.sub(*probs.topk(2, dim = dim).values)

def compute_output_lengths(x, lengths_fraction):
	return (lengths_fraction * x.shape[-1] + 0.5).int()

def compute_capacity(model):
	return sum(p.numel() for p in model.parameters())
