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
			super().__init__(nn.Conv1d(input_size, num_classes[0], kernel_size = 1))
		elif type == 'bpe':
			super().__init__(
				nn.Conv1d(input_size, num_classes[0], kernel_size = 1), 
				nn.Sequential(
					#nn.Conv1d(input_size, num_classes[1], kernel_size = 1)#, padding = 7)
					ConvBN(num_channels = (input_size, input_size), kernel_size = 15), 
					ConvBN(num_channels = (input_size, num_classes[1]), kernel_size = 15)
				)
			)
		self.type = type

	def forward(self, x):
		if self.type is None:
			return self[0](x)
		elif self.type == 'bpe':
			y1 = self[0](x)
			y2 = self[1](x)
			return y1, y2

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
	def __init__(self, num_channels, kernel_size, stride = 1, dropout = 0, batch_norm_momentum = 0.1, groups = 1, num_channels_residual = [], repeat = 1, dilation = 1, separable = False, temporal_mask = True, inplace = False, nonlinearity = 'relu', residual = False):
		super().__init__()
		num_channels_residual = num_channels_residual or ([None] if residual else [])
		self.conv = nn.ModuleList(ConvSamePadding(num_channels[0] if i == 0 else num_channels[1], num_channels[1], kernel_size = kernel_size, stride = stride, dilation = dilation, separable = separable, bias = False, groups = groups) for i in range(repeat))
		self.bn = nn.ModuleList(BatchNorm1dInplace(num_channels[1], momentum = batch_norm_momentum) if inplace else nn.BatchNorm1d(num_channels[1], momentum = batch_norm_momentum) for i in range(repeat))
		self.conv_residual = nn.ModuleList(nn.Identity() if in_channels is None else nn.Conv1d(in_channels, num_channels[1], kernel_size = 1) for in_channels in num_channels_residual)
		self.bn_residual = nn.ModuleList(nn.Identity() if in_channels is None else BatchNorm1dInplace(num_channels[1], momentum = batch_norm_momentum) if inplace else nn.BatchNorm1d(num_channels[1], momentum = batch_norm_momentum) for in_channels in num_channels_residual)
		self.activation = ResidualActivation(nonlinearity, dropout, inplace = inplace)
		self.temporal_mask = temporal_mask

	def forward(self, x, lengths_fraction = None, residual = []):
		residual = residual or [x]
		for i, (conv, bn) in enumerate(zip(self.conv, self.bn)):
			x = self.activation(bn(conv(x)), residual = [bn(conv(r)) for conv, bn, r in zip(self.conv_residual, self.bn_residual, residual)] if i == len(self.conv) - 1 else [])
			x = x * temporal_mask(x, lengths_fraction = lengths_fraction) if (self.temporal_mask and lengths_fraction is not None) else x
		return x

	def fuse_conv_bn_eval(self):
		for i in range(len(self.conv_residual)):
			conv, bn = self.conv_residual[i], self.bn_residual[i]
			if type(conv) != nn.Identity and type(bn) != nn.Identity:
				self.conv_residual[i] = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
				self.bn_residual[i] = nn.Identity()

		for i in range(len(self.conv)):
			conv, bn = self.conv[i][-1], self.bn[i]
			self.conv[i][-1] = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
			self.bn[i] = nn.Identity()

# Jasper 5x3: 5 blocks, each has 1 sub-blocks, each sub-block has 3 ConvBnRelu
# Jasper 10x5: 5 blocks, each has 2 sub-blocks, each sub-block has 5 ConvBnRelu
# residual = 'dense' | True | False
class JasperNet(nn.ModuleList):
	def __init__(self, num_input_features, num_classes, repeat = 3, num_subblocks = 1, dilation = 1, residual = 'dense',
			kernel_sizes = [11, 13, 17, 21, 25], kernel_size_prologue = 11, kernel_size_epilogue = 29, 
			base_width = 128, out_width_factors = [2, 3, 4, 5, 6], out_width_factors_large = [7, 8],
			separable = False, groups = 1, 
			dropout = 0, dropout_prologue = 0.2, dropout_epilogue = 0.4, dropouts = [0.2, 0.2, 0.2, 0.3, 0.3],
			temporal_mask = True, nonlinearity = 'relu', inplace = False,
			stride1 = 2, stride2 = 1, decoder_type = None
		):
		dropout_prologue = dropout_prologue if dropout != 0 else 0
		dropout_epilogue = dropout_epilogue if dropout != 0 else 0
		dropouts = dropouts if dropout != 0 else [0] * len(dropouts)

		in_width_factor = 2
		prologue = [ConvBN(kernel_size = kernel_size_prologue, num_channels = (num_input_features, in_width_factor * base_width), dropout = dropout_prologue, stride = stride1, temporal_mask = temporal_mask, nonlinearity = nonlinearity, inplace = inplace)]
		
		backbone = []
		num_channels_residual = []
		for kernel_size, dropout, out_width_factor in zip(kernel_sizes, dropouts, out_width_factors):
			for s in range(num_subblocks):
				num_channels = (in_width_factor * base_width, (out_width_factor * base_width) if s == num_subblocks - 1 else (in_width_factor * base_width))
				#num_channels = (in_width_factor * base_wdith, out_width_factor * base_width) # seems they do this in https://github.com/NVIDIA/DeepLearningExamples/blob/21120850478d875e9f2286d13143f33f35cd0c74/PyTorch/SpeechRecognition/Jasper/configs/jasper10x5dr_nomask.toml
				num_channels_residual.append(in_width_factor * base_width)
				# use None in num_channels_residual
				backbone.append(ConvBN(kernel_size = kernel_size, num_channels = num_channels, dropout = dropout, repeat = repeat, separable = separable, groups = groups, num_channels_residual = num_channels_residual, temporal_mask = temporal_mask, nonlinearity = nonlinearity, inplace = inplace,))
			in_width_factor = out_width_factor

		epilogue = [
			ConvBN(kernel_size = kernel_size_epilogue, num_channels = (in_width_factor * base_width, out_width_factors_large[0] * base_width), dropout = dropout_epilogue, dilation = dilation, temporal_mask = temporal_mask, nonlinearity = nonlinearity, inplace = inplace),
			ConvBN(kernel_size = 1, num_channels = (out_width_factors_large[0] * base_width, out_width_factors_large[1] * base_width), dropout = dropout_epilogue, temporal_mask = temporal_mask, nonlinearity = nonlinearity, inplace = inplace),
		]
		decoder = [
			Decoder(out_width_factors_large[1] * base_width, num_classes, type = decoder_type)
		]
		super().__init__(prologue + backbone + epilogue + decoder)
		self.residual = residual

	def forward(self, x, xlen, y = None, ylen = None):
		residual = []
		for i, subblock in enumerate(list(self)[:-1]):
			x = subblock(x, residual = residual if i < len(self) - 3 else [], lengths_fraction = xlen)
			if self.residual != 'dense':
				residual.clear()
			if self.residual:
				residual.append(x)

		#logits_char, logits_bpe = self[-1](x)

		logits_char = self[-1](x)
		log_probs_char = F.log_softmax(logits_char, dim = 1)
		output_lengths_char = compute_output_lengths(log_probs_char, xlen)
		
		if y is not None and ylen is not None:
			loss_char = F.ctc_loss(log_probs_char.permute(2, 0, 1), y[:, 0], output_lengths_char, ylen[:, 0], blank = log_probs_char.shape[1] - 1, reduction = 'none') / ylen[:, 0]
			loss = loss_char
			#loss_bpe =  F.ctc_loss(log_probs_bpe.permute(2, 0, 1) , y[:, 1], output_lengths_bpe,  ylen[:, 1], blank = log_probs_bpe.shape[ 1] - 1, reduction = 'none') / ylen[:, 1]
			#output_lengths_bpe = compute_output_lengths(log_probs_bpe, xlen)
			#log_probs_bpe = F.log_softmax(logits_bpe, dim = 1)
			
			#loss = loss_char + loss_bpe
		else:
			loss = torch.tensor(0.)

		#return dict(log_probs = [log_probs_char, log_probs_bpe], output_lengths = [output_lengths_char, output_lengths_bpe], loss = loss, loss_ = dict(char = loss_char, bpe = loss_bpe))
		
		return dict(log_probs = [log_probs_char], output_lengths = [output_lengths_char], loss = loss)

	def freeze(self, backbone = True, decoder0 = True):
		for m in (list(self)[:-1] if backbone else []) + (list(self[-1])[:1] if decoder0 else []):
			for module in filter(lambda module: isinstance(module, nn.modules.batchnorm._BatchNorm), m.modules()):
				module.eval()
				module.train = lambda training: None

			for p in m.parameters():
				p.requires_grad = False

	def fuse_conv_bn_eval(self):
		for block in list(self)[:-1]:
			block.fuse_conv_bn_eval()

class Wav2Letter(JasperNet):
	def __init__(self, num_input_features, num_classes, dropout = 0.2, nonlinearity = ('hardtanh', 0, 20), kernel_size_prologue = 11, kernel_size_epilogue = 29, kernel_sizes = [11, 13, 17, 21, 25], dilation = 2):
		super().__init__(num_input_features, num_classes, base_width = base_width, 
			dropout = dropout, dropout_prologue = dropout, dropout_epilogue = dropout, dropouts = [dropout] * num_blocks, 
			kernel_size_prologue = kernel_size_prologue, kernel_size_epilogue = kernel_size_epilogue, kernel_sizes = [kernel_size_prologue] * num_blocks,
			out_width_factors = [2, 3, 4, 5, 6], out_width_factors_large = [7, 8], 
			residual = False, diletion = dilation, nonlinearity = nonlinearity
		)
		
class Wav2LetterFlat(JasperNet):
	def __init__(self, num_input_features, num_classes, dropout = 0.2, base_width = 128, width_factor_large = 16, width_factor = 6, kernel_size_epilogue = 29, kernel_size_prologue = 13, num_blocks = 6):
		super().__init__(num_input_features, num_classes, base_width = base_width, 
			dropout = dropout, dropout_prologue = dropout, dropout_epilogue = dropout, dropouts = [dropout] * num_blocks, 
			kernel_size_prologue = kernel_size_prologue, kernel_size_epilogue = kernel_size_epilogue, kernel_sizes = [kernel_size_prologue] * num_blocks,
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

class JasperNetBigInplaceLargeStride(JasperNet):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, num_subblocks = 2, temporal_mask = False, inplace = True, nonlinearity = ('leaky_relu', 0.01), dilation = 2, **kwargs)

class ResidualActivation(nn.Module):
	def __init__(self, nonlinearity, dropout = 0, inplace = False):
		super().__init__()
		self.nonlinearity = nonlinearity
		self.inplace = inplace
		self.dropout = dropout

	def forward(self, y, residual = []):
		assert not (self.inplace and self.nonlinearity == 'relu')

		if self.inplace:
			y = ResidualActivation.Function.apply(self.nonlinearity, y, *residual)
			y = F.dropout(y, p = self.dropout, training = self.training)
		else:
			y = y + sum(residual)
			if self.nonlinearity == 'relu':
				y = relu_dropout(y, p = self.dropout, inplace = True, training = self.training) # F.dropout(F.relu(y, inplace = True), p = self.dropout, training = self.training)
			elif self.nonlinearity and self.nonlinearity[0] in ['leaky_relu', 'hardtanh']:
				y = F.dropout(getattr(F, self.nonlinearity[0])(y, *self.nonlinearity[1:], inplace = True), p = self.dropout, training = self.training)
		return y

	class Function(torch.autograd.function.Function):
		@staticmethod
		def forward(self, nonlinearity, x, *residual):
			self.nonlinearity = nonlinearity
			x_ = x.data
			for r in residual:
				x_ += r
			if self.nonlinearity and self.nonlinearity[0] == 'leaky_relu':
				F.leaky_relu_(x_, self.nonlinearity[1])
			self.save_for_backward(x, *residual)
			return x

		@staticmethod
		def backward(self, grad_output):
			x, *residual = self.saved_tensors
			x_ = x.data
			if self.nonlinearity and self.nonlinearity[0] == 'leaky_relu':
				mask = torch.ones_like(grad_output).masked_fill_(x < 0, self.nonlinearity[1])
				grad_output *= mask
				x_ /= mask
			for r in residual:
				x_ -= r
			return (None, ) + (grad_output,) * (1 + len(residual))

class BatchNorm1dInplace(nn.BatchNorm1d):
	def forward(self, input):
		return BatchNorm1dInplace.Function.apply(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.momentum, self.training) 

	class Function(torch.autograd.function.Function):
		@staticmethod
		def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, training):
			mean, var = torch.batch_norm_update_stats(input, running_mean, running_var, momentum) if training else (running_mean, running_var) 
			invstd = (var + eps).rsqrt_()
			output = torch.batch_norm_elemt(input, weight, bias, mean, invstd, 0, out = input)
			self.save_for_backward(output, weight, bias, mean, invstd)
			return output

		@staticmethod
		def backward(self, grad_output):
			saved_output, weight, bias, mean, invstd = self.saved_tensors
			saved_input = torch.batch_norm_elemt(saved_output, invstd.reciprocal(), mean, bias, weight.reciprocal(), 0, out = saved_output)
			mean_dy, mean_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(grad_output, saved_input, mean, invstd,	weight,	*self.needs_input_grad[:3])
			grad_input = torch.batch_norm_backward_elemt(grad_output, saved_input, mean, invstd, weight, mean_dy, mean_dy_xmu)
			return grad_input, grad_weight, grad_bias, None, None, None, None, None

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

def unpad(x, lens, device = 'cpu'):
	return [e[..., :l].to(device) for e, l in zip(x, lens)]
