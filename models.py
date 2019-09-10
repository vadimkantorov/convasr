import os
import collections
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

class MBConvBlock(nn.Module):
	def __init__(self, kernel_size, num_channels, stride = 1, dilation = 1, dropout = 0.2, repeat = 1, expansion = 1, squeeze_excitation_ratio = 0.25, separable = False, padding = None, simple = False, batch_norm_momentum = 0.1):
		super().__init__()
		in_channels, out_channels = num_channels
		padding = padding if padding is not None else max(1, dilation * (kernel_size // 2))

		self.subblocks = nn.ModuleList()
		for k in range(repeat):
			in_channels_ = in_channels if k == 0 else out_channels
			exp_channels = in_channels * expansion if not simple else out_channels
			se_channels = int(exp_channels * squeeze_excitation_ratio)
			groups = exp_channels if separable else 1

			self.subblocks.append(nn.ModuleDict(dict(
				expand = nn.Sequential(
					nn.Conv1d(in_channels_, exp_channels, kernel_size, stride = stride, bias = False, padding = padding, groups = groups, dilation = dilation),
					nn.BatchNorm1d(exp_channels, momentum = batch_norm_momentum),
					ReLUDropout(p = dropout, inplace = True)
				),

				squeeze_and_excite = nn.Sequential(
					nn.Conv1d(exp_channels, se_channels, kernel_size = 1),
					nn.ReLU(inplace = True),
					nn.Conv1d(se_channels, exp_channels, kernel_size = 1),
					nn.Sigmoid()
				) if not simple else nn.Identity(),

				reduce = nn.Sequential(
					nn.Conv1d(exp_channels, out_channels, bias = False, stride = 1, kernel_size = 1),
					nn.BatchNorm1d(out_channels, momentum = batch_norm_momentum),
				) if not simple else nn.Identity()
			)))

		self.residual = nn.Identity() if in_channels == out_channels or simple else nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size = 1, stride = stride), nn.BatchNorm1d(out_channels, momentum = batch_norm_momentum))
		self.simple = simple
	
	def forward(self, x):
		if self.simple:
			return self.subblocks[0].expand(x)

		y = x
		for subblock in self.subblocks:
			y = subblock.expand(y)
			y = y * subblock.squeeze_and_excite(y)
			y = subblock.reduce(y)
		return y + self.residual(x) if x.shape[-1] == y.shape[-1] else y

class BabbleNet(nn.Sequential):
	def __init__(self, num_classes, num_input_features, dropout = 0.2, repeat = 1, batch_norm_momentum = 0.1):
			super().__init__(
				MBConvBlock(kernel_size = 11, num_channels = (num_input_features, 256), dropout = dropout, stride = 2, simple = True),

				MBConvBlock(kernel_size = 11, num_channels = (256, 256), dropout = dropout, repeat = repeat),
				MBConvBlock(kernel_size = 13, num_channels = (256, 384), dropout = dropout, repeat = repeat),
				MBConvBlock(kernel_size = 17, num_channels = (384, 512), dropout = dropout, repeat = repeat),
				MBConvBlock(kernel_size = 21, num_channels = (512, 640), dropout = dropout, repeat = repeat),
				MBConvBlock(kernel_size = 25, num_channels = (640, 768), dropout = dropout, repeat = repeat),
					
				MBConvBlock(kernel_size = 29, num_channels = (768, 896), dropout = 0.4, dilation = 2),
				MBConvBlock(kernel_size = 1, num_channels = (896, 1024), dropout = 0.4),
				nn.Conv1d(1024, num_classes, kernel_size = 1)
			)

	def forward(self, x, input_lengths):
		return super().forward(x)

#TODO: apply conv masking
class Wav2LetterRu(nn.Sequential):
	def __init__(self, num_classes, num_input_features, dropout = [0.2] * 9):
		dropout_ = lambda i: dropout[i] if isinstance(dropout, list) else dropout
		def conv_bn_relu_dropout(kernel_size, num_channels, stride, dropout, padding = None, batch_norm_momentum = 0.1):
			return nn.Sequential(collections.OrderedDict(zip(['0', '1', '2'], [ #['conv', 'bn', 'relu_dropout'], [
				nn.Conv1d(num_channels[0], num_channels[1], kernel_size = kernel_size, stride = stride, padding = padding if padding is not None else max(1, kernel_size // 2), bias = False),
				nn.BatchNorm1d(num_channels[1], momentum = batch_norm_momentum),
				ReLUDropout(p = dropout, inplace = True)
				]))
			)

		layers = [
			conv_bn_relu_dropout(kernel_size = 13, num_channels = (num_input_features, 768), stride = 2, dropout = dropout_(0)),
			conv_bn_relu_dropout(kernel_size = 13, num_channels = (768, 768), stride = 1, dropout = dropout_(1)),
			conv_bn_relu_dropout(kernel_size = 13, num_channels = (768, 768), stride = 1, dropout = dropout_(2)),
			conv_bn_relu_dropout(kernel_size = 13, num_channels = (768, 768), stride = 1, dropout = dropout_(3)),
			conv_bn_relu_dropout(kernel_size = 13, num_channels = (768, 768), stride = 1, dropout = dropout_(4)),
			conv_bn_relu_dropout(kernel_size = 13, num_channels = (768, 768), stride = 1, dropout = dropout_(5)),
			conv_bn_relu_dropout(kernel_size = 13, num_channels = (768, 768), stride = 1, dropout = dropout_(6)),
			conv_bn_relu_dropout(kernel_size = 31, num_channels = (768, 2048), stride = 1, dropout = dropout_(7)),
			conv_bn_relu_dropout(kernel_size = 1,  num_channels = (2048, 2048), stride = 1, dropout = dropout_(8)),
			nn.Conv1d(2048, num_classes, kernel_size = 1, stride = 1)
		]
		super().__init__(*layers)

	def forward(self, x, input_lengths):
		return super().forward(x)

class Wav2LetterVanilla(nn.Sequential):
	def __init__(self, num_classes, num_input_features):
		def conv_bn_clip(kernel_size, num_channels, stride = 1, dilation = 1, repeat = 1, padding = 0):
			modules = []
			for i in range(repeat):
				modules.append(nn.Conv1d(num_channels[0] if i == 0 else num_channels[1], num_channels[1], kernel_size = kernel_size, stride = stride, dilation = dilation, padding = padding))
				modules.append(nn.Hardtanh(0, 20, inplace = True))
			return nn.Sequential(*modules)

		layers = [
			conv_bn_clip(kernel_size = 11, num_channels = (num_input_features, 256), stride = 2, padding = 5), # 64
			conv_bn_clip(kernel_size = 11, num_channels = (256, 256), repeat = 3, padding = 5),
			conv_bn_clip(kernel_size = 13, num_channels = (256, 384), repeat = 3, padding = 6),
			conv_bn_clip(kernel_size = 17, num_channels = (384, 512), repeat = 3, padding = 8),
			conv_bn_clip(kernel_size = 21, num_channels = (512, 640), repeat = 3, padding = 10),
			conv_bn_clip(kernel_size = 25, num_channels = (640, 768), repeat = 3, padding = 12),
			conv_bn_clip(kernel_size = 29, num_channels = (768, 896), repeat = 1, padding = 28, dilation = 2),
			conv_bn_clip(kernel_size = 1, num_channels = (896, 1024), repeat = 1),
			nn.Conv1d(1024, num_classes, kernel_size = 1)
		]

		super().__init__(*layers)

class JasperNet(nn.ModuleList):
	def __init__(self, num_classes, num_input_features, repeat = 3, num_subblocks = 1):
		def conv_bn_relu_dropout_residual(kernel_size, num_channels, dropout = 0, stride = 1, dilation = 1, padding = 0, batch_norm_momentum = 0.1, repeat = 1, num_channels_residual = []):
				return nn.ModuleDict(dict(
					relu_dropout = ReLUDropout(p = dropout, inplace = True),
					conv = nn.ModuleList([nn.Conv1d(num_channels[0] if i == 0 else num_channels[1], num_channels[1], kernel_size = kernel_size, stride = stride, dilation = dilation, padding = padding, bias = False) for i in range(repeat)]),
					bn = nn.ModuleList([nn.BatchNorm1d(num_channels[1], momentum = batch_norm_momentum) for i in range(repeat)]),
					conv_residual = nn.ModuleList([nn.Conv1d(in_channels, num_channels[1], kernel_size = 1) for in_channels in num_channels_residual]),
					bn_residual = nn.ModuleList([nn.BatchNorm1d(num_channels[1], momentum = batch_norm_momentum) for in_channels in num_channels_residual])
				))

		prologue = [conv_bn_relu_dropout_residual(kernel_size = 11, num_channels = (num_input_features, 256), dropout = 0.2, padding = 5, stride = 2)]
		backbone = [
				conv_bn_relu_dropout_residual(kernel_size = 11, num_channels = (256, 256), dropout = 0.2, padding = 5, repeat = repeat, num_channels_residual = [256]),
				conv_bn_relu_dropout_residual(kernel_size = 11, num_channels = (256, 256), dropout = 0.2, padding = 5, repeat = repeat, num_channels_residual = [256, 256]),
				conv_bn_relu_dropout_residual(kernel_size = 13, num_channels = (256, 384), dropout = 0.2, padding = 6, repeat = repeat, num_channels_residual = [256, 256, 256]),
				conv_bn_relu_dropout_residual(kernel_size = 13, num_channels = (384, 384), dropout = 0.2, padding = 6, repeat = repeat, num_channels_residual = [256, 256, 256, 384]),
				conv_bn_relu_dropout_residual(kernel_size = 17, num_channels = (384, 512), dropout = 0.2, padding = 8, repeat = repeat, num_channels_residual = [256, 256, 256, 384, 384]),
				conv_bn_relu_dropout_residual(kernel_size = 17, num_channels = (512, 512), dropout = 0.2, padding = 8, repeat = repeat, num_channels_residual = [256, 256, 256, 384, 384, 512]),
				conv_bn_relu_dropout_residual(kernel_size = 21, num_channels = (512, 640), dropout = 0.3, padding = 10, repeat = repeat, num_channels_residual = [256, 256, 256, 384, 384, 512, 512]),
				conv_bn_relu_dropout_residual(kernel_size = 21, num_channels = (640, 640), dropout = 0.3, padding = 10, repeat = repeat, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640]),
				conv_bn_relu_dropout_residual(kernel_size = 25, num_channels = (640, 768), dropout = 0.3, padding = 12, repeat = repeat, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640, 640]),
				conv_bn_relu_dropout_residual(kernel_size = 25, num_channels = (768, 768), dropout = 0.3, padding = 12, repeat = repeat, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640, 640, 768]),
			] if num_subblocks == 2 else [
				conv_bn_relu_dropout_residual(kernel_size = 11, num_channels = (256, 256), dropout = 0.2, padding = 5,  repeat = repeat, num_channels_residual = [256]),
				conv_bn_relu_dropout_residual(kernel_size = 13, num_channels = (256, 384), dropout = 0.2, padding = 6,  repeat = repeat, num_channels_residual = [256, 256]),
				conv_bn_relu_dropout_residual(kernel_size = 17, num_channels = (384, 512), dropout = 0.2, padding = 8,  repeat = repeat, num_channels_residual = [256, 256, 384]),
				conv_bn_relu_dropout_residual(kernel_size = 21, num_channels = (512, 640), dropout = 0.3, padding = 10, repeat = repeat, num_channels_residual = [256, 256, 384, 512]),
				conv_bn_relu_dropout_residual(kernel_size = 25, num_channels = (640, 768), dropout = 0.3, padding = 12, repeat = repeat, num_channels_residual = [256, 256, 384, 512, 640])
			]
		epilogue = [
			conv_bn_relu_dropout_residual(kernel_size = 29, num_channels = (768, 896), dropout = 0.4, padding = 28, dilation = 2),
			conv_bn_relu_dropout_residual(kernel_size = 1, num_channels = (896, 1024), dropout = 0.4),

			nn.Conv1d(1024, num_classes, kernel_size = 1)
		]
		super().__init__(prologue + backbone + epilogue)

	def forward(self, x):
		residual = []
		for i, block in enumerate(list(self)[:-1]):
			for conv, bn in zip(block.conv[:-1], block.bn[:-1]):
				x = bn(conv(x))
				x = block.relu_dropout(x)
			x = block.bn[-1](block.conv[-1](x))
			for conv, bn, r in zip(block.conv_residual, block.bn_residual, residual if i < len(self) - 3 else []):
				x = x + bn(conv(r))
			x = block.relu_dropout(x)
			residual.append(x)
		return self[-1](x)

def compute_output_lengths(model, input_lengths):
	return input_lengths.int() // 2

def compute_capacity(model):
	return sum(p.numel() for p in model.parameters())

class ReLUDropout(torch.nn.Module):
	def __init__(self, p, inplace):
		super().__init__()
		self.p = p

	def forward(self, input):
		if self.training:
			p1m = 1. - self.p
			mask = torch.rand_like(input) < p1m
			mask *= (input > 0)
			return input.masked_fill_(~mask, 0).mul_(1.0 / p1m)
		else:
			return input.clamp_(min = 0)

class MaskedConv1d(nn.Conv1d):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, use_mask=True):
		super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
		self.use_mask = use_mask

		def get_seq_len(self, lens):
			return ((lens + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)

		def forward(self, inp):
			x, lens = inp
			if self.use_mask:
				max_len = x.size(2)
				mask = torch.arange(max_len, dtype = lens.dtype, device = lens.device).expand(len(lens), max_len) >= lens.unsqueeze(1)
				x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
				del mask
				lens = self.get_seq_len(lens)
			out = super().forward(x)
			return out, lens

def logfbank(signal, sample_rate, window_size, window_stride, window, num_input_features, dither = 1e-5, preemph = 0.97, normalize = True, eps = 1e-20):
	signal = normalize_signal(signal)
	signal = torch.cat([signal[:1], signal[1:] - preemph * signal[:-1]])
	win_length, hop_length = int(window_size * sample_rate), int(window_stride * sample_rate)
	n_fft = 2 ** math.ceil(math.log2(win_length))
	signal += dither * torch.randn_like(signal)
	window = getattr(torch, window)(win_length, periodic = False).type_as(signal)
	mel_basis = torch.from_numpy(librosa.filters.mel(sample_rate, n_fft, n_mels=num_input_features, fmin=0, fmax=int(sample_rate/2))).type_as(signal)
	power_spectrum = torch.stft(signal, n_fft, hop_length = hop_length, win_length = win_length, window = window, pad_mode = 'reflect', center = True).pow(2).sum(dim = -1)
	features = torch.log(torch.matmul(mel_basis, power_spectrum) + eps)
	return normalize_features(features) if normalize else features 

def normalize_signal(signal, eps = 1e-5):
	signal = signal.to(torch.float32)
	return signal / (signal.abs().max() + eps)

def normalize_features(features, eps = 1e-20):
	return (features - features.mean(dim = -1, keepdim = True)) / (features.std(dim = -1, keepdim = True) + eps)

def temporal_mask(features, lengths):
	mask = torch.ones_like(features)
	for m, l in zip(mask, lengths):
		m[..., l:].zero_()
	return mask

def entropy(log_probs, lengths, dim = 1, eps = 1e-9):
	e = (log_probs.exp() * log_probs).sum(dim = dim)
	return -(e * temporal_mask(e, lengths)).sum(dim = -1) / (eps + lengths.type_as(log_probs))
