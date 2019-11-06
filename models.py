import os
import math
import collections
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

class Decoder(nn.Sequential):
	def __init__(self, input_size, num_classes, type = None):
		if type is None:
			super().__init__(nn.Conv1d(input_size, num_classes, kernel_size = 1))
		elif type == 'gru':
			super().__init__(nn.Conv1d(input_size, num_classes, kernel_size = 1), nn.GRU(num_classes, num_classes, batch_first = True, bidirectional = False))
		elif type == 'transformerencoder':
			super().__init__(nn.Conv1d(input_size, num_classes, kernel_size = 1), nn.TransformerEncoderLayer(num_classes, nhead = 2, dim_feedforward = num_classes))
		self.type = type

	def forward(self, x):
		if self.type is None:
			y = self[0](x)
		elif self.type == 'gru':
			y = self[0](x)
			#y = self[1](y.transpose(-1, -2))[0].transpose(-1, -2) # + y
		elif self.type == 'transformerencoder':
			y = self[0](x)
			y = y + self[1](y.transpose(-1, -2)).transpose(-1, -2)
		return y

class ConvSamePadding(nn.Sequential):
	def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, bias, groups, separable):
		padding = dilation * max(1, kernel_size // 2)
		if separable:
			assert dilation == 1
			super().__init__(
				nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups),
				nn.ReLU(inplace = True),
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
		self.conv_residual = nn.ModuleList(nn.Conv1d(in_channels, num_channels[1], kernel_size = 1) if in_channels is not None else nn.Identity() for in_channels in num_channels_residual)
		self.bn_residual = nn.ModuleList(ActivatedBatchNorm(num_channels[1], momentum = batch_norm_momentum, nonlinearity = None, inplace = inplace) if in_channels is not None else nn.Identity() for in_channels in num_channels_residual)
		self.temporal_mask = temporal_mask

	def forward(self, x, lengths_fraction = None, residual = []):
		y = x
		for i, (conv, bn) in enumerate(zip(self.conv, self.bn)):
			y = bn(conv(y), residual = [bn(conv(r)) for conv, bn, r in zip(self.conv_residual, self.bn_residual, residual)] if i == len(self.conv) - 1 else [])
			y = y * temporal_mask(y, lengths_fraction = lengths_fraction) if (self.temporal_mask and lengths_fraction is not None) else y
		return y

class JasperNet(nn.ModuleList):
	def __init__(self, num_classes, num_input_features, repeat = 3, num_subblocks = 1, dilation = 1, residual = 'dense',
			kernel_sizes = [11, 13, 17, 21, 25], kernel_size_small = 11, kernel_size_large = 29, 
			base_width = 128, out_width_factors = [2, 3, 4, 5, 6], out_width_factors_large = [7, 8],
			separable = False, groups = 1, 
			dropout = None, dropout_small = 0.2, dropout_large = 0.4, dropouts = [0.2, 0.2, 0.2, 0.3, 0.3],
			temporal_mask = True, nonlinearity = 'relu', inplace = False
		):
		dropout_small = dropout_small if dropout != 0 else 0
		dropout_large = dropout_large if dropout != 0 else 0
		dropouts = dropouts if dropout != 0 else [0] * len(dropouts)

		width_factor_ = 2
		prologue = [ConvBN(kernel_size = kernel_size_small, num_channels = (num_input_features, width_factor_ * base_width), dropout = dropout_small, stride = 2, temporal_mask = temporal_mask, nonlinearity = nonlinearity, inplace = inplace)]
		
		backbone = []
		num_channels_residual = []
		for kernel_size, dropout, width_factor in zip(kernel_sizes, dropouts, out_width_factors):
			for s in range(num_subblocks):
				num_channels = (width_factor_ * base_width, (width_factor * base_width) if s == num_subblocks - 1 else (width_factor_ * base_width))
				num_channels_residual.append(width_factor_ * base_width)
				# use None in num_channels_residual
				backbone.append(ConvBN(kernel_size = kernel_size, num_channels = num_channels, dropout = dropout, repeat = repeat, separable = separable, groups = groups, num_channels_residual = num_channels_residual, temporal_mask = temporal_mask, nonlinearity = nonlinearity, inplace = inplace))
			width_factor_ = width_factor

		epilogue = [
			ConvBN(kernel_size = kernel_size_large, num_channels = (width_factor_ * base_width, out_width_factors_large[0] * base_width), dropout = dropout_large, dilation = dilation, temporal_mask = temporal_mask, nonlinearity = nonlinearity, inplace = inplace),
			ConvBN(kernel_size = 1, num_channels = (out_width_factors_large[0] * base_width, out_width_factors_large[1] * base_width), dropout = dropout_large, temporal_mask = temporal_mask, nonlinearity = nonlinearity, inplace = inplace),
		]
		decoder = [
			Decoder(out_width_factors_large[1] * base_width, num_classes, type = None)
		]
		super().__init__(prologue + backbone + epilogue + decoder)
		self.residual = residual

	def forward(self, x, lengths_fraction):
		residual = []
		for i, subblock in enumerate(list(self)[:-1]):
			x = subblock(x, residual = residual if i < len(self) - 3 else [], lengths_fraction = lengths_fraction)
			if self.residual != 'dense':
				residual.clear()
			if self.residual:
				residual.append(x)

		logits = self[-1](x)
		return logits, compute_output_lengths(logits, lengths_fraction)

class Wav2Letter(JasperNet):
	def __init__(self, num_classes, num_input_features, dropout = 0.2, nonlinearity = ('hardtanh', 0, 20), kernel_size_small = 11, kernel_size_large = 29, kernel_sizes = [11, 13, 17, 21, 25], dilation = 2):
		super().__init__(num_classes, num_input_features, base_width = base_width, 
			dropout = dropout, dropout_small = dropout, dropout_large = dropout, dropouts = [dropout] * num_blocks, 
			kernel_size_small = kernel_size_small, kernel_size_large = kernel_size_large, kernel_sizes = [kernel_size_small] * num_blocks,
			out_width_factors = [2, 3, 4, 5, 6], out_width_factors_large = [7, 8], 
			residual = False, diletion = dilation, nonlinearity = nonlinearity
		)
		

class Wav2LetterFlat(JasperNet):
	def __init__(self, num_classes, num_input_features, dropout = 0.2, base_width = 128, width_factor_large = 16, width_factor = 6, kernel_size_large = 29, kernel_size_small = 13, num_blocks = 6):
		super().__init__(num_classes, num_input_features, base_width = base_width, 
			dropout = dropout, dropout_small = dropout, dropout_large = dropout, dropouts = [dropout] * num_blocks, 
			kernel_size_small = kernel_size_small, kernel_size_large = kernel_size_large, kernel_sizes = [kernel_size_small] * num_blocks,
			out_width_factors = [width_factor] * num_blocks, out_width_factors_large = [width_factor_large, width_factor_large], 
			residual = False
		)

class JasperNetSeparable(JasperNet):
	def __init__(self, *args, separable = True, groups = 128, **kwargs):
		super().__init__(*args, separable = separable, groups = groups, **kwargs)

class JasperNetBig(JasperNet):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, num_subblocks = 2, temporal_mask = False, **kwargs)

class JasperNetBigInplace(JasperNet):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, num_subblocks = 2, temporal_mask = False, inplace = True, nonlinearity = ('leaky_relu', 0.01), **kwargs)

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
			y = self.squeeze_and_excite(y) if self.squeeze_and_excite is not None else y
			y = y + sum(residual)
			if self.nonlinearity == 'relu':
				y = relu_dropout(y, p = self.dropout, inplace = True, training = self.training)
			elif self.nonlinearity and self.nonlinearity[0] in ['leaky_relu', 'hardtanh']:
				y = F.dropout(getattr(F, self.nonlinearity[0])(y, *self.nonlinearity[1:], inplace = True), p = self.dropout, training = self.training)
		return y

	class Function(torch.autograd.function.Function):
		@staticmethod
		def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, training, nonlinearity, *residual):
			self.nonlinearity = nonlinearity
			assert input.is_contiguous()
			
			mean, var = torch.batch_norm_update_stats(input, running_mean, running_var, momentum) if training else (running_mean, running_var) 
			invstd = (var + eps).rsqrt_()
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

			saved_input = torch.batch_norm_elemt(saved_output, saved_output, invstd.reciprocal(), mean, bias, weight.reciprocal(), 0)
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

	def forward(self, x):
		return x * super().forward(x)

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
