import os
import math
import collections
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import apex
import librosa
import shaping
import typing

class InputOutputTypeCast(nn.Module):
	def __init__(self, model, dtype):
		super().__init__()
		self.model = model
		self.dtype = dtype

	def forward(self, x, *args, **kwargs):
		return self.model(x.to(self.dtype), *args, **kwargs).to(x.dtype)


class Decoder(nn.Sequential):
	def __init__(self, input_size, num_classes, type = None):
		if type is None:
			super().__init__(nn.Conv1d(input_size, num_classes[0], kernel_size = 1))
		elif type == 'bpe':
			super().__init__(
				nn.Conv1d(input_size, num_classes[0], kernel_size = 1),
				nn.Sequential(
					#nn.Conv1d(input_size, num_classes[1], kernel_size = 1)#, padding = 7)
					ConvBn1d(num_channels = (input_size, input_size), kernel_size = 15),
					ConvBn1d(num_channels = (input_size, num_classes[1]), kernel_size = 15)
				)
			)
		self.type = type

	def forward(self, x):
		if self.type is None:
			return (self[0](x), )
		elif self.type == 'bpe':
			y1 = self[0](x)
			y2 = self[1](x)
			return y1, y2


class ConvSamePadding(nn.Sequential):
	def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, bias, groups, separable):
		padding = dilation * kernel_size // 2
		if separable:
			assert dilation == 1
			super().__init__(
				nn.Conv1d(
					in_channels,
					out_channels,
					kernel_size = kernel_size,
					stride = stride,
					padding = padding,
					dilation = dilation,
					groups = groups
				),
				nn.ReLU(inplace = True),
				nn.Conv1d(out_channels, out_channels, kernel_size = 1, bias = bias)
			)
		else:
			super().__init__(
				nn.Conv1d(
					in_channels,
					out_channels,
					kernel_size = kernel_size,
					stride = stride,
					padding = padding,
					dilation = dilation,
					groups = groups,
					bias = bias
				)
			)


class ConvBn1d(nn.Module):
	def __init__(
		self,
		num_channels,
		kernel_size,
		stride = 1,
		dropout = 0,
		groups = 1,
		num_channels_residual: typing.List = [],
		repeat = 1,
		dilation = 1,
		separable = False,
		temporal_mask = True,
		nonlinearity = ('relu', ),
		nonlinearity_reference = True,
		batch_norm_momentum = 0.1,
		inplace = False
	):
		super().__init__()
		self.conv = nn.ModuleList(
			ConvSamePadding(
				num_channels[0] if i == 0 else num_channels[1],
				num_channels[1],
				kernel_size = kernel_size,
				stride = stride,
				dilation = dilation,
				separable = separable,
				bias = False,
				groups = groups
			) for i in range(repeat)
		)
		self.bn = nn.ModuleList(
			InplaceBatchNorm1d(num_channels[1], momentum = batch_norm_momentum) if inplace else nn
			.BatchNorm1d(num_channels[1], momentum = batch_norm_momentum) for i in range(repeat)
		)
		self.conv_residual = nn.ModuleList(
			nn.Identity() if in_channels is None else nn.Conv1d(in_channels, num_channels[1], kernel_size = 1)
			for in_channels in num_channels_residual
		)
		self.bn_residual = nn.ModuleList(
			nn.Identity() if in_channels is None else
			InplaceBatchNorm1d(num_channels[1], momentum = batch_norm_momentum) if inplace else nn
			.BatchNorm1d(num_channels[1], momentum = batch_norm_momentum) for in_channels in num_channels_residual
		)
		self.activation = ResidualActivation(nonlinearity, dropout, invertible = inplace)
		self.temporal_mask = temporal_mask

	def forward(self, x, lengths_fraction = None, residual: typing.List = []):
		for i, (conv, bn) in enumerate(zip(self.conv, self.bn)):
			if i == len(self.conv) - 1:
				assert len(self.conv_residual) == len(self.bn_residual) == len(residual)
				residual_inputs = [bn(conv(r)) for conv, bn, r in zip(self.conv_residual, self.bn_residual, residual)]
			else:
				residual_inputs = []

			x = self.activation(bn(conv(x)), residual = residual_inputs)
			if self.temporal_mask and lengths_fraction is not None:
				lengths = compute_output_lengths(x, lengths_fraction)
				x = x * temporal_mask(x, lengths)
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


#TODO: figure out perfect same padding
# Jasper 5x3: 5 blocks, each has 1 sub-blocks, each sub-block has 3 ConvBnRelu
# Jasper 10x5: 5 blocks, each has 2 sub-blocks, each sub-block has 5 ConvBnRelu
# residual = 'dense' | True | False
class JasperNet(nn.Module):
	def __init__(
		self,
		num_input_features,
		num_classes,
		repeat = 3,
		num_subblocks = 1,
		dilation = 1,
		residual = 'dense',
		kernel_sizes = [11, 13, 17, 21, 25],
		kernel_size_prologue = 11,
		kernel_size_epilogue = 29,
		base_width = 128,
		out_width_factors = [2, 3, 4, 5, 6],
		out_width_factors_large = [7, 8],
		separable = False,
		groups = 1,
		dropout = 0,
		dropout_prologue = 0.2,
		dropout_epilogue = 0.4,
		dropouts = [0.2, 0.2, 0.2, 0.3, 0.3],
		temporal_mask = True,
		nonlinearity = ('relu', ),
		inplace = False,
		stride1 = 2,
		stride2 = 1,
		decoder_type = None,
		dict = dict,
		frontend = None,
		bpe_only = False,
		normalize_features = True,
		normalize_features_eps = torch.finfo(torch.float16).tiny,
		normalize_features_track_running_stats = False,
		normalize_features_legacy = True,
		normalize_features_temporal_mask = True
	):
		super().__init__()
		self.init_params = {name: repr(value) for name, value in locals().items()}
		dropout_prologue = dropout_prologue if dropout != 0 else 0
		dropout_epilogue = dropout_epilogue if dropout != 0 else 0
		dropouts = dropouts if dropout != 0 else [0] * len(dropouts)

		in_width_factor = out_width_factors[0]
		self.backbone = nn.ModuleList([
			ConvBn1d(
				kernel_size = kernel_size_prologue,
				num_channels = (num_input_features, in_width_factor * base_width),
				dropout = dropout_prologue,
				stride = stride1,
				temporal_mask = temporal_mask,
				nonlinearity = nonlinearity,
				inplace = inplace
			)
		])
		num_channels_residual = []
		for kernel_size, dropout, out_width_factor in zip(kernel_sizes, dropouts, out_width_factors):
			for s in range(num_subblocks):
				num_channels = (
					in_width_factor * base_width, (out_width_factor * base_width) if s == num_subblocks - 1 else
					(in_width_factor * base_width)
				)
				#num_channels = (in_width_factor * base_wdith, out_width_factor * base_width) # seems they do this in https://github.com/NVIDIA/DeepLearningExamples/blob/21120850478d875e9f2286d13143f33f35cd0c74/PyTorch/SpeechRecognition/Jasper/configs/jasper10x5dr_nomask.toml
				if residual == 'dense':
					num_channels_residual.append(num_channels[0])
				elif residual == 'flat':
					num_channels_residual = [None]
				elif residual:
					num_channels_residual = [num_channels[0]]
				else:
					num_channels_residual = []
				self.backbone.append(
					ConvBn1d(
						num_channels = num_channels,
						kernel_size = kernel_size,
						dropout = dropout,
						repeat = repeat,
						separable = separable,
						groups = groups,
						num_channels_residual = num_channels_residual,
						temporal_mask = temporal_mask,
						nonlinearity = nonlinearity,
						inplace = inplace
					)
				)
			in_width_factor = out_width_factor

		epilogue = [
			ConvBn1d(
				num_channels = (in_width_factor * base_width, out_width_factors_large[0] * base_width),
				kernel_size = kernel_size_epilogue,
				dropout = dropout_epilogue,
				dilation = dilation,
				temporal_mask = temporal_mask,
				nonlinearity = nonlinearity,
				inplace = inplace
			),
			ConvBn1d(
				num_channels = (out_width_factors_large[0] * base_width, out_width_factors_large[1] * base_width),
				kernel_size = 1,
				dropout = dropout_epilogue,
				temporal_mask = temporal_mask,
				nonlinearity = nonlinearity,
				inplace = inplace
			),
		]
		self.backbone.extend(epilogue)

		self.num_epilogue_modules = len(epilogue)
		self.frontend = frontend
		self.normalize_features = MaskedInstanceNorm1d(
			num_input_features,
			affine = False,
			eps = normalize_features_eps,
			track_running_stats = normalize_features_track_running_stats,
			temporal_mask = normalize_features_temporal_mask,
			legacy = normalize_features_legacy
		) if normalize_features else None
		self.decoder = Decoder(out_width_factors_large[1] * base_width, num_classes, type = decoder_type)
		self.residual = residual
		self.dict = dict
		self.bpe_only = bpe_only

	def forward(
		self, x: typing.Union[shaping.BCT, shaping.BT], xlen: typing.Optional[shaping.B] = None, y: typing.Optional[shaping.BY] = None, ylen: typing.Optional[shaping.B] = None
	) -> shaping.BCt:

		if self.frontend is not None:
			lengths = compute_output_lengths(x, xlen)
			x = self.frontend(x.squeeze(dim = 1), mask = temporal_mask(x, lengths))

		assert x.ndim == 3

		if self.normalize_features is not None:
			lengths = compute_output_lengths(x, xlen)
			x = self.normalize_features(x, mask = temporal_mask(x, lengths))

		residual = []
		for i, subblock in enumerate(self.backbone):
			x = subblock(x, residual = residual, lengths_fraction = xlen)
			if i >= len(self.backbone) - self.num_epilogue_modules - 1:  # HACK: drop residual connections for epilogue
				residual = []
			elif self.residual == 'dense':
				residual.append(x)
			elif self.residual:
				residual = [x]
			else:
				residual = []

		logits = self.decoder(x)
		log_probs = [F.log_softmax(l, dim = 1).to(torch.float32) for l in logits]
		olen = [compute_output_lengths(l, xlen.to(torch.float32) if xlen is not None else None) for l in logits]
		aux = {}

		if y is not None and ylen is not None:
			loss = []
			for i, l in enumerate(log_probs):
				loss.append(F.ctc_loss(l.permute(2, 0, 1), y[:, i], olen[i], ylen[:, i], blank = l.shape[1] - 1, reduction = 'none') / ylen[:, 0])
			aux = dict(loss = sum(loss) if not self.bpe_only else sum(loss[1:]))

		return self.dict(logits = logits, log_probs = log_probs, olen = olen, **aux)

	def freeze(self, backbone = 0, decoder0 = False, frontend = False):
		frozen_modules_list = []
		frozen_modules_list += list(self.backbone[:backbone]) if backbone else []
		frozen_modules_list += (list(self.decoder)[:1] if decoder0 else [])
		frozen_modules_list += ([self.frontend] if frontend and self.frontend is not None else [])

		for m in frozen_modules_list:
			for module in filter(lambda module: isinstance(module, nn.modules.batchnorm._BatchNorm), m.modules()):
				module.eval()
				module.train = lambda training: None
			for p in m.parameters():
				p.requires_grad = False

	def fuse_conv_bn_eval(self, K = None):
		for subblock in self.backbone[:K]:
			subblock.fuse_conv_bn_eval()

	def set_temporal_mask_mode(self, enabled):
		for module in self.modules():
			module.temporal_mask = enabled


class ResidualActivation(nn.Module):
	def __init__(self, nonlinearity, dropout = 0, invertible = False):
		super().__init__()
		self.nonlinearity = nonlinearity
		self.dropout = dropout
		self.invertible = invertible

	def forward(self, y, residual: typing.List = []):
		if self.invertible:
			y = ResidualActivation.InvertibleResidualInplaceFunction.apply(self.nonlinearity, y, *residual)
			y = F.dropout(y, p = self.dropout, training = self.training)
		else:
			for r in residual:
				y += r

			if self.dropout > 0 and self.training and self.nonlinearity[0] == 'relu':
				y = relu_dropout(y, p = self.dropout, training = self.training, inplace = True)
			else:
				y = getattr(F, self.nonlinearity[0])(y, *self.nonlinearity[1:], inplace = True)
				y = F.dropout(y, p = self.dropout, training = self.training)

		return y

	def extra_repr(self):
		return f'nonlinearity={self.nonlinearity}, dropout={self.dropout}, invertible={self.invertible}'

	class InvertibleResidualInplaceFunction(torch.autograd.function.Function):
		@staticmethod
		def forward(ctx, nonlinearity, x, *residual):
			ctx.nonlinearity = nonlinearity
			assert ctx.nonlinearity and ctx.nonlinearity[0] in ['leaky_relu']

			y = x.data
			for r in residual:
				y += r
			y = getattr(F, ctx.nonlinearity[0])(y, *ctx.nonlinearity[1:], inplace = True)
			#ctx.mark_dirty(x)
			ctx.save_for_backward(x, *residual)
			return x

		@staticmethod
		def backward(ctx, grad_output):
			x, *residual = ctx.saved_tensors
			y = x.data
			mask = torch.ones_like(grad_output).masked_fill_(x < 0, ctx.nonlinearity[1])
			grad_output.mul_(mask)  #grad_output = grad_output.contiguous().mul_(mask)
			y /= mask
			for r in residual:
				y -= r
			return (None, ) + (grad_output, ) * (1 + len(residual))


class InplaceBatchNorm1d(nn.BatchNorm1d):
	def forward(self, input):
		return InplaceBatchNorm1d.Function.apply(
			input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.momentum, self.training
		)

	class Function(torch.autograd.function.Function):
		@staticmethod
		def forward(ctx, input, weight, bias, running_mean, running_var, eps, momentum, training):
			mean, var = torch.batch_norm_update_stats(input, running_mean, running_var, momentum) if training else (running_mean, running_var)
			invstd = (var + eps).rsqrt_()
			output = torch.batch_norm_elemt(input, weight, bias, mean, invstd, 0, out = input)
			ctx.training = training
			ctx.save_for_backward(input, weight, bias, mean, invstd)
			ctx.mark_dirty(input)
			return input

		@staticmethod
		def backward(ctx, grad_output):
			assert ctx.training, 'Inplace BatchNorm supports only training mode, input gradients with running stats used are not yet supported'
			saved_output, weight, bias, mean, invstd = ctx.saved_tensors
			saved_input = torch.batch_norm_elemt(
				saved_output, invstd.reciprocal(), mean, bias, weight.reciprocal(), 0, out = saved_output
			)
			sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(grad_output, saved_input, mean, invstd, weight, *ctx.needs_input_grad[:3])
			divisor = saved_input.numel() // saved_input.size(1)
			mean_dy = sum_dy.div_(divisor)
			mean_dy_xmu = sum_dy_xmu.div_(divisor)
			grad_input = torch.batch_norm_backward_elemt(
				grad_output, saved_input, mean, invstd, weight, mean_dy, mean_dy_xmu
			)
			return grad_input, grad_weight, grad_bias, None, None, None, None, None


def relu_dropout(x, p = 0, inplace = False, training = False):
	if not training or p == 0:
		return x.clamp_(min = 0) if inplace else x.clamp(min = 0)

	p1m = 1 - p
	mask = torch.rand_like(x) > p1m
	mask |= (x < 0)
	return x.masked_fill_(mask, 0).div_(p1m) if inplace else x.masked_fill(mask, 0).div(p1m)


class AugmentationFrontend(nn.Module):
	def __init__(self, frontend, feature_transform = None, waveform_transform = None):
		super().__init__()
		self.frontend = frontend
		self.feature_transform = feature_transform
		self.waveform_transform = waveform_transform

	def forward(self, signal, audio_path = None, dataset_name = None, waveform_transform_debug = None, **kwargs):
		if self.waveform_transform is not None:
			signal = self.waveform_transform(signal, self.frontend.sample_rate, dataset_name = dataset_name)

		if waveform_transform_debug is not None:
			waveform_transform_debug(audio_path, self.frontend.sample_rate, signal)

		features = self.frontend(signal)

		if self.feature_transform is not None:
			features = self.feature_transform(features, self.frontend.sample_rate, dataset_name = dataset_name)

		return features

	@property
	def sample_rate(self):
		return self.frontend.sample_rate

	@property
	def read_audio(self):
		return 'SoxAug' not in self.waveform_transform.__class__.__name__


class Wav2VecFrontend(nn.Module):
	def __init__(
		self, out_channels, sample_rate, preemphasis = 0.0, use_context_features = True, extra_args = None, **kwargs
	):
		from fairseq.models.wav2vec import Wav2VecModel

		assert sample_rate == extra_args.sample_rate, f'Sample rate {sample_rate} is not equal to frontend sample rate {extra_args.sample_rate}, use --sample-rate {extra_args.sample_rate}'

		if extra_args.aggregator == 'cnn':
			agg_layers = eval(extra_args.conv_aggregator_layers)
			agg_dim = agg_layers[-1][0]
			assert out_channels == agg_dim, f'Out channels {out_channels} is not equal to frontend output dim {agg_dim}, use --num-input-features {agg_dim}'
		elif extra_args.aggregator == 'gru':
			assert out_channels == extra_args.gru_dim, f'Out channels {out_channels} is not equal to frontend output dim {extra_args.gru_dim}, use --num-input-features {extra_args.gru_dim}'
		else:
			raise RuntimeError(f'Wrong wav2vec aggregator {extra_args.aggregator}. Use cnn or gru instead.')

		super().__init__()
		self.fairseq_args = extra_args
		self.preemphasis = preemphasis
		self.use_context_features = use_context_features
		self.model = Wav2VecModel.build_model(extra_args, None).eval()

	def forward(self, signal: shaping.BT, mask: shaping.BT = None) -> shaping.BCT:
		assert signal.is_floating_point(), f'Wrong signal type {signal.dtype}, use normalized floating point signal instead.'
		signal = torch.cat([signal[..., :1], signal[..., 1:] -
							self.preemphasis * signal[..., :-1]], dim = -1) if self.preemphasis > 0 else signal
		signal = signal * mask if mask is not None else signal

		raw_features = self.model.feature_extractor(signal)
		if isinstance(raw_features, tuple):
			raw_features = raw_features[0]

		if self.use_context_features:
			context_features = self.model.feature_aggregator(raw_features)
			return context_features
		else:
			return raw_features


class LogFilterBankFrontend(nn.Module):
	def __init__(
		self,
		out_channels,
		sample_rate,
		window_size,
		window_stride,
		window,
		dither = 1e-5,
		dither0 = 0.0,
		preemphasis = 0.97,
		eps = torch.finfo(torch.float16).tiny,
		normalize_signal = True,
		stft_mode = None,
		window_periodic = True,
		normalize_features = False,
		**kwargs
	):
		super().__init__()
		self.stft_mode = stft_mode
		self.dither = dither
		self.dither0 = dither0
		self.preemphasis = preemphasis
		self.normalize_signal = normalize_signal
		self.sample_rate = sample_rate

		self.win_length = int(window_size * sample_rate)
		self.hop_length = int(window_stride * sample_rate)
		self.nfft = 2**math.ceil(math.log2(self.win_length))
		self.freq_cutoff = self.nfft // 2 + 1

		self.register_buffer('window', getattr(torch, window)(self.win_length, periodic = window_periodic).float())
		#mel_basis = torchaudio.functional.create_fb_matrix(n_fft, n_mels = num_input_features, fmin = 0, fmax = int(sample_rate/2)).t() # when https://github.com/pytorch/audio/issues/287 is fixed
		mel_basis = torch.as_tensor(
			librosa.filters.mel(sample_rate, self.nfft, n_mels = out_channels, fmin = 0, fmax = int(sample_rate / 2))
		)
		self.mel = nn.Conv1d(mel_basis.shape[1], mel_basis.shape[0], 1).requires_grad_(False)
		self.mel.weight.copy_(mel_basis.unsqueeze(-1))
		self.mel.bias.fill_(eps)

		if stft_mode == 'conv':
			fourier_basis = torch.rfft(torch.eye(self.nfft), signal_ndim = 1, onesided = False)
			forward_basis = fourier_basis[:self.freq_cutoff].permute(2, 0, 1).reshape(-1, 1, fourier_basis.shape[1])
			forward_basis = forward_basis * torch.as_tensor(
				librosa.util.pad_center(self.window, self.nfft), dtype = forward_basis.dtype
			)
			self.stft = nn.Conv1d(
				forward_basis.shape[1],
				forward_basis.shape[0],
				forward_basis.shape[2],
				bias = False,
				stride = self.hop_length
			).requires_grad_(False)
			self.stft.weight.copy_(forward_basis)
		else:
			self.stft = None

	def forward(self, signal: shaping.BT, mask: shaping.BT = None) -> shaping.BCT:
		assert signal.ndim == 2

		signal = signal if signal.is_floating_point() else signal.to(torch.float32)
		signal = normalize_signal(signal) if self.normalize_signal else signal
		signal = apply_dither(signal, self.dither0)
		signal = torch.cat([signal[..., :1], signal[..., 1:] -
							self.preemphasis * signal[..., :-1]], dim = -1) if self.preemphasis > 0 else signal
		signal = apply_dither(signal, self.dither)
		signal = signal * mask if mask is not None else signal

		pad = self.freq_cutoff - 1
		padded_signal = F.pad(signal.unsqueeze(1), (pad, 0), mode = 'reflect').squeeze(
			1
		)  # TODO: remove un/squeeze when https://github.com/pytorch/pytorch/issues/29863 is fixed
		padded_signal = F.pad(
			padded_signal, (0, pad), mode = 'constant', value = 0
		)  # TODO: avoid this second copy by doing pad manually

		real_squared, imag_squared = self.stft(padded_signal.unsqueeze(dim = 1)).pow(2).split(self.freq_cutoff, dim = 1) if self.stft is not None else padded_signal.stft(self.nfft, hop_length = self.hop_length, win_length = self.win_length, window = self.window, center = False).pow(2).unbind(dim = -1)
		power_spectrum = real_squared + imag_squared
		log_mel_features = self.mel(power_spectrum).log()
		return log_mel_features

	@property
	def read_audio(self):
		return True

@torch.jit.script
def compute_output_lengths(x: shaping.BT, lengths_fraction: typing.Optional[shaping.B] = None):
	if lengths_fraction is None:
		return torch.full(x.shape[:1], x.shape[-1], device = x.device, dtype = torch.long)
	else:
		return (lengths_fraction * x.shape[-1]).ceil().long()

@torch.jit.script
def temporal_mask(x: shaping.BT, lengths: shaping.B):
	return (torch.arange(x.shape[-1], device = x.device, dtype = lengths.dtype).unsqueeze(0) <
			lengths.unsqueeze(1)).view(x.shape[:1] + (1, ) * (len(x.shape) - 2) + x.shape[-1:])

@torch.jit.script
def apply_dither(x: shaping.BT, dither: float):
	# dither extracted to ScriptFunction, because JIT does not trace randn_like correctly https://github.com/pytorch/pytorch/issues/43767
	if dither > 0.0:
		return x + dither * torch.randn_like(x)
	else:
		return x

def entropy(log_probs, lengths = None, dim = 1, eps = 1e-9, sum = True, keepdim = False):
	e = -(log_probs.exp() * log_probs).sum(dim = dim, keepdim = keepdim)
	if lengths is not None:
		e = e * temporal_mask(e, lengths)
	return (
		e.sum(dim = -1) / (eps + lengths.type_as(log_probs)) if lengths is not None else e.mean(dim = -1)
	) if sum else e


def margin(log_probs, dim = 1):
	return torch.sub(*log_probs.exp().topk(2, dim = dim).values)

def compute_capacity(model, scale = 1):
	return sum(map(torch.numel, model.parameters())) / scale


def normalize_signal(signal, dim = -1, eps = 1e-5):
	return signal / (signal.abs().max(dim = dim, keepdim = True).values + eps) if signal.numel() > 0 else signal


class MaskedInstanceNorm1d(nn.InstanceNorm1d):
	def __init__(self, *args, temporal_mask = False, legacy = True, **kwargs):
		super().__init__(*args, **kwargs)
		self.temporal_mask = temporal_mask
		self.legacy = legacy

	def forward(self, x, mask = None):
		if not self.temporal_mask or mask is None:
			if self.legacy:
				assert self.track_running_stats is False
				std, mean = torch.std_mean(x, dim = -1, keepdim = True)
				return (x - mean) / (std + self.eps)
			else:
				return super().forward(x)
		else:
			assert self.track_running_stats is False
			xlen = mask.int().sum(dim = -1, keepdim = True)
			mean = (x * mask).sum(dim = -1, keepdim = True) / xlen
			zero_mean_masked = mask * (x - mean)
			std = (zero_mean_masked.pow(2).sum(dim = -1, keepdim = True) / xlen).sqrt()
			return zero_mean_masked / (std + self.eps)


def unpad(x, lens):
	return [e[..., :l] for e, l in zip(x, lens)]


def reset_bn_running_stats_(model):
	for bn in [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]:
		nn.init.zeros_(bn.running_mean)
		nn.init.ones_(bn.running_var)
		nn.init.zeros_(bn.num_batches_tracked)
		bn.momentum = None
		bn.train()
	return model


def data_parallel_and_autocast(model, optimizer = None, data_parallel = True, opt_level = None, **kwargs):
	data_parallel = data_parallel and torch.cuda.device_count() > 1

	if opt_level is None:
		model = torch.nn.DataParallel(model) if data_parallel else model

	elif data_parallel:
		model, optimizer = apex.amp.initialize(nn.Sequential(model), optimizers = optimizer, opt_level = opt_level, **kwargs) if optimizer is not None else (apex.amp.initialize(nn.Sequential(model), opt_level = opt_level, **kwargs), None)
		model = torch.nn.DataParallel(model[0])
		model.forward = lambda *args, old_fwd = model.forward, input_caster = lambda tensor: tensor.to(apex.amp._amp_state.opt_properties.options['cast_model_type']) if tensor.is_floating_point() else tensor, output_caster = lambda tensor: (tensor.to(apex.amp._amp_state.opt_properties.options['cast_model_outputs'] if apex.amp._amp_state.opt_properties.options.get('cast_model_outputs') is not None else torch.float32)) if tensor.is_floating_point() else tensor, **kwargs: apex.amp._initialize.applier(old_fwd(*apex.amp._initialize.applier(args, input_caster), **apex.amp._initialize.applier(kwargs, input_caster)), output_caster)

	else:
		model, optimizer = apex.amp.initialize(model, optimizers = optimizer, opt_level = opt_level, **kwargs) if optimizer is not None else (apex.amp.initialize(model, opt_level = opt_level, **kwargs), None)

	return model, optimizer


def silence_space_mask(log_probs, speech, blank_idx, space_idx, kernel_size = 101):
	# major dilation
	greedy_decoded = log_probs.max(dim = 1).indices
	silence = ~speech & (greedy_decoded == blank_idx)
	return silence[:, None, :] * (
		~F.one_hot(torch.tensor(space_idx), log_probs.shape[1]).to(device = silence.device, dtype = silence.dtype)
	)[None, :, None]

def rle1d(tensor):
	starts = torch.cat(( torch.LongTensor([0], device = tensor.device), (tensor[1:] != tensor[:-1]).nonzero(as_tuple = False).add_(1).squeeze(1), torch.LongTensor([tensor.shape[-1]], device = tensor.device) ))
	starts, lengths, values = starts[:-1], (starts[1:] - starts[:-1]), tensor[starts[:-1]]
	return starts, lengths, values


def sparse_topk(x, k, dim = -1, largest = True, indices_dtype = None, values_dtype = None, fill_value = 0.0):
	topk = x.topk(k, dim = dim, largest = largest)
	return dict(
		k = k,
		dim = dim,
		largest = largest,
		shape = x.shape,
		dtype = x.dtype,
		device = x.device,
		fill_value = fill_value,
		indices = topk.indices.to(dtype = indices_dtype),
		values = topk.values.to(dtype = values_dtype)
	)


def sparse_topk_todense(saved, device = None):
	device = device or saved['device']
	return torch.full(saved['shape'], saved['fill_value'], dtype = saved['dtype'], device = device).scatter_(
		saved['dim'],
		saved['indices'].to(dtype = torch.int64, device = device),
		saved['values'].to(dtype = saved['dtype'], device = device)
	)


def master_module(model):
	return model.module if isinstance(model, nn.DataParallel) else model


########CONFIGS########


class Wav2Letter(JasperNet):
	def __init__(
		self,
		num_input_features,
		num_classes,
		dropout = 0.2,
		base_width = 128,
		nonlinearity = ('hardtanh', 0, 20),
		kernel_size_prologue = 11,
		kernel_size_epilogue = 29,
		kernel_sizes = [11, 13, 17, 21, 25],
		dilation = 2,
		num_blocks = 6,
		decoder_type = None,
		normalize_features = True,
		frontend = None
	):
		super().__init__(
			num_input_features,
			num_classes,
			base_width = base_width,
			dropout = dropout,
			dropout_prologue = dropout,
			dropout_epilogue = dropout,
			dropouts = [dropout] * num_blocks,
			kernel_size_prologue = kernel_size_prologue,
			kernel_size_epilogue = kernel_size_epilogue,
			kernel_sizes = [kernel_size_prologue] * num_blocks,
			out_width_factors = [2, 3, 4, 5, 6],
			out_width_factors_large = [7, 8],
			residual = False,
			dilation = dilation,
			nonlinearity = nonlinearity,
			decoder_type = decoder_type,
			normalize_features = normalize_features,
			frontend = frontend
		)


class Wav2LetterResidual(JasperNet):
	def __init__(
		self,
		num_input_features,
		num_classes,
		dropout = 0.2,
		base_width = 128,
		nonlinearity = ('hardtanh', 0, 20),
		kernel_size_prologue = 11,
		kernel_size_epilogue = 29,
		kernel_sizes = [11, 13, 17, 21, 25],
		dilation = 2,
		num_blocks = 5,
		decoder_type = None,
		normalize_features = True,
		frontend = None
	):
		super().__init__(
			num_input_features,
			num_classes,
			base_width = base_width,
			dropout = dropout,
			dropout_prologue = dropout,
			dropout_epilogue = dropout,
			dropouts = [dropout] * num_blocks,
			kernel_size_prologue = kernel_size_prologue,
			kernel_size_epilogue = kernel_size_epilogue,
			kernel_sizes = [kernel_size_prologue] * num_blocks,
			out_width_factors = [2, 3, 4, 5, 6],
			out_width_factors_large = [7, 8],
			residual = True,
			dilation = dilation,
			nonlinearity = nonlinearity,
			decoder_type = decoder_type,
			normalize_features = normalize_features,
			frontend = frontend
		)


class Wav2LetterResidualNoDilation(JasperNet):
	def __init__(
		self,
		num_input_features,
		num_classes,
		dropout = 0.2,
		base_width = 128,
		nonlinearity = ('hardtanh', 0, 20),
		kernel_size_prologue = 11,
		kernel_size_epilogue = 29,
		kernel_sizes = [11, 13, 17, 21, 25],
		dilation = 1,
		num_blocks = 5,
		decoder_type = None,
		normalize_features = True,
		frontend = None
	):
		super().__init__(
			num_input_features,
			num_classes,
			base_width = base_width,
			dropout = dropout,
			dropout_prologue = dropout,
			dropout_epilogue = dropout,
			dropouts = [dropout] * num_blocks,
			kernel_size_prologue = kernel_size_prologue,
			kernel_size_epilogue = kernel_size_epilogue,
			kernel_sizes = [kernel_size_prologue] * num_blocks,
			out_width_factors = [2, 3, 4, 5, 6],
			out_width_factors_large = [7, 8],
			residual = True,
			dilation = dilation,
			nonlinearity = nonlinearity,
			decoder_type = decoder_type,
			normalize_features = normalize_features,
			frontend = frontend
		)


class Wav2LetterResidualBig(JasperNet):
	def __init__(
		self,
		num_input_features,
		num_classes,
		dropout = 0.2,
		base_width = 128,
		nonlinearity = ('hardtanh', 0, 20),
		kernel_size_prologue = 11,
		kernel_size_epilogue = 29,
		kernel_sizes = [11, 13, 17, 21, 25],
		dilation = 2,
		num_blocks = 5,
		decoder_type = None,
		normalize_features = True,
		frontend = None
	):
		super().__init__(
			num_input_features,
			num_classes,
			base_width = base_width,
			num_subblocks = 2,
			dropout = dropout,
			dropout_prologue = dropout,
			dropout_epilogue = dropout,
			dropouts = [dropout] * num_blocks,
			kernel_size_prologue = kernel_size_prologue,
			kernel_size_epilogue = kernel_size_epilogue,
			kernel_sizes = [kernel_size_prologue] * num_blocks,
			out_width_factors = [2, 3, 4, 5, 6],
			out_width_factors_large = [7, 8],
			residual = True,
			dilation = dilation,
			nonlinearity = nonlinearity,
			decoder_type = decoder_type,
			normalize_features = normalize_features,
			frontend = frontend
		)


class Wav2LetterDense(JasperNet):
	def __init__(
		self,
		num_input_features,
		num_classes,
		dropout = 0.2,
		base_width = 128,
		nonlinearity = ('hardtanh', 0, 20),
		kernel_size_prologue = 11,
		kernel_size_epilogue = 29,
		kernel_sizes = [11, 13, 17, 21, 25],
		dilation = 2,
		num_blocks = 5,
		decoder_type = None,
		normalize_features = True,
		frontend = None
	):
		super().__init__(
			num_input_features,
			num_classes,
			base_width = base_width,
			dropout = dropout,
			dropout_prologue = dropout,
			dropout_epilogue = dropout,
			dropouts = [dropout] * num_blocks,
			kernel_size_prologue = kernel_size_prologue,
			kernel_size_epilogue = kernel_size_epilogue,
			kernel_sizes = [kernel_size_prologue] * num_blocks,
			out_width_factors = [2, 3, 4, 5, 6],
			out_width_factors_large = [7, 8],
			residual = 'dense',
			dilation = dilation,
			nonlinearity = nonlinearity,
			decoder_type = decoder_type,
			normalize_features = normalize_features,
			frontend = frontend
		)


class Wav2LetterDenseNoDilation(JasperNet):
	def __init__(
		self,
		num_input_features,
		num_classes,
		dropout = 0.2,
		base_width = 128,
		nonlinearity = ('hardtanh', 0, 20),
		kernel_size_prologue = 11,
		kernel_size_epilogue = 29,
		kernel_sizes = [11, 13, 17, 21, 25],
		dilation = 1,
		num_blocks = 5,
		decoder_type = None,
		normalize_features = True,
		frontend = None
	):
		super().__init__(
			num_input_features,
			num_classes,
			base_width = base_width,
			dropout = dropout,
			dropout_prologue = dropout,
			dropout_epilogue = dropout,
			dropouts = [dropout] * num_blocks,
			kernel_size_prologue = kernel_size_prologue,
			kernel_size_epilogue = kernel_size_epilogue,
			kernel_sizes = [kernel_size_prologue] * num_blocks,
			out_width_factors = [2, 3, 4, 5, 6],
			out_width_factors_large = [7, 8],
			residual = 'dense',
			dilation = dilation,
			nonlinearity = nonlinearity,
			decoder_type = decoder_type,
			normalize_features = normalize_features,
			frontend = frontend
		)


class Wav2LetterDenseNoDilationInplace(JasperNet):
	def __init__(
		self,
		num_input_features,
		num_classes,
		dropout = 0.2,
		base_width = 128,
		nonlinearity = ('leaky_relu', 0.01),
		kernel_size_prologue = 11,
		kernel_size_epilogue = 29,
		kernel_sizes = [11, 13, 17, 21, 25],
		dilation = 1,
		num_blocks = 5,
		decoder_type = None,
		normalize_features = True,
		frontend = None
	):
		super().__init__(
			num_input_features,
			num_classes,
			base_width = base_width,
			inplace = True,
			dropout = dropout,
			dropout_prologue = dropout,
			dropout_epilogue = dropout,
			dropouts = [dropout] * num_blocks,
			kernel_size_prologue = kernel_size_prologue,
			kernel_size_epilogue = kernel_size_epilogue,
			kernel_sizes = [kernel_size_prologue] * num_blocks,
			out_width_factors = [2, 3, 4, 5, 6],
			out_width_factors_large = [7, 8],
			residual = 'dense',
			dilation = dilation,
			nonlinearity = nonlinearity,
			decoder_type = decoder_type,
			normalize_features = normalize_features,
			frontend = frontend
		)


class Wav2LetterDenseLargeKernels(JasperNet):
	def __init__(
		self,
		num_input_features,
		num_classes,
		dropout = 0.2,
		base_width = 128,
		nonlinearity = ('hardtanh', 0, 20),
		kernel_size_prologue = 11,
		kernel_size_epilogue = 29,
		kernel_sizes = [11, 13, 17, 21, 25],
		dilation = 2,
		num_blocks = 5,
		decoder_type = None,
		normalize_features = True,
		frontend = None
	):
		super().__init__(
			num_input_features,
			num_classes,
			base_width = base_width,
			dropout = dropout,
			dropout_prologue = dropout,
			dropout_epilogue = dropout,
			dropouts = [dropout] * num_blocks,
			kernel_size_prologue = kernel_size_prologue,
			kernel_size_epilogue = kernel_size_epilogue,
			kernel_sizes = kernel_sizes,
			out_width_factors = [2, 3, 4, 5, 6],
			out_width_factors_large = [7, 8],
			residual = 'dense',
			dilation = dilation,
			nonlinearity = nonlinearity,
			decoder_type = decoder_type,
			normalize_features = normalize_features,
			frontend = frontend
		)


class Wav2LetterDenseNoDilationLargeKernels(JasperNet):
	def __init__(
		self,
		num_input_features,
		num_classes,
		dropout = 0.2,
		base_width = 128,
		nonlinearity = ('hardtanh', 0, 20),
		kernel_size_prologue = 11,
		kernel_size_epilogue = 29,
		kernel_sizes = [11, 13, 17, 21, 25],
		dilation = 1,
		num_blocks = 5,
		decoder_type = None,
		normalize_features = True,
		frontend = None
	):
		super().__init__(
			num_input_features,
			num_classes,
			base_width = base_width,
			dropout = dropout,
			dropout_prologue = dropout,
			dropout_epilogue = dropout,
			dropouts = [dropout] * num_blocks,
			kernel_size_prologue = kernel_size_prologue,
			kernel_size_epilogue = kernel_size_epilogue,
			kernel_sizes = kernel_sizes,
			out_width_factors = [2, 3, 4, 5, 6],
			out_width_factors_large = [7, 8],
			residual = 'dense',
			dilation = dilation,
			nonlinearity = nonlinearity,
			decoder_type = decoder_type,
			normalize_features = normalize_features,
			frontend = frontend
		)


class Wav2LetterDenseBig(JasperNet):
	def __init__(
		self,
		num_input_features,
		num_classes,
		dropout = 0.2,
		base_width = 128,
		nonlinearity = ('hardtanh', 0, 20),
		kernel_size_prologue = 11,
		kernel_size_epilogue = 29,
		kernel_sizes = [11, 13, 17, 21, 25],
		dilation = 2,
		num_blocks = 5,
		decoder_type = None,
		normalize_features = True,
		frontend = None
	):
		super().__init__(
			num_input_features,
			num_classes,
			base_width = base_width,
			num_subblocks = 2,
			dropout = dropout,
			dropout_prologue = dropout,
			dropout_epilogue = dropout,
			dropouts = [dropout] * num_blocks,
			kernel_size_prologue = kernel_size_prologue,
			kernel_size_epilogue = kernel_size_epilogue,
			kernel_sizes = [kernel_size_prologue] * num_blocks,
			out_width_factors = [2, 3, 4, 5, 6],
			out_width_factors_large = [7, 8],
			residual = 'dense',
			dilation = dilation,
			nonlinearity = nonlinearity,
			decoder_type = decoder_type,
			normalize_features = normalize_features,
			frontend = frontend
		)


class Wav2LetterDenseBigLargeKernelsNoDropoutReLu(JasperNet):
	def __init__(
		self,
		num_input_features,
		num_classes,
		dropout = 0.0,
		base_width = 128,
		nonlinearity = ('relu', ),
		kernel_size_prologue = 11,
		kernel_size_epilogue = 29,
		kernel_sizes = [11, 13, 17, 21, 25],
		dilation = 2,
		num_blocks = 5,
		decoder_type = None,
		normalize_features = True,
		frontend = None
	):
		super().__init__(
			num_input_features,
			num_classes,
			base_width = base_width,
			num_subblocks = 2,
			dropout = dropout,
			dropout_prologue = dropout,
			dropout_epilogue = dropout,
			dropouts = [dropout] * num_blocks,
			kernel_size_prologue = kernel_size_prologue,
			kernel_size_epilogue = kernel_size_epilogue,
			kernel_sizes = kernel_sizes,
			out_width_factors = [2, 3, 4, 5, 6],
			out_width_factors_large = [7, 8],
			residual = 'dense',
			dilation = dilation,
			nonlinearity = nonlinearity,
			decoder_type = decoder_type,
			normalize_features = normalize_features,
			frontend = frontend
		)


class Wav2LetterDenseBigLargeKernelsNoDilationNoDropoutReLu(JasperNet):
	def __init__(
		self,
		num_input_features,
		num_classes,
		dropout = 0.0,
		base_width = 128,
		nonlinearity = ('relu', ),
		kernel_size_prologue = 11,
		kernel_size_epilogue = 29,
		kernel_sizes = [11, 13, 17, 21, 25],
		dilation = 1,
		num_blocks = 5,
		decoder_type = None,
		normalize_features = True,
		frontend = None
	):
		super().__init__(
			num_input_features,
			num_classes,
			base_width = base_width,
			num_subblocks = 2,
			dropout = dropout,
			dropout_prologue = dropout,
			dropout_epilogue = dropout,
			dropouts = [dropout] * num_blocks,
			kernel_size_prologue = kernel_size_prologue,
			kernel_size_epilogue = kernel_size_epilogue,
			kernel_sizes = kernel_sizes,
			out_width_factors = [2, 3, 4, 5, 6],
			out_width_factors_large = [7, 8],
			residual = 'dense',
			dilation = dilation,
			nonlinearity = nonlinearity,
			decoder_type = decoder_type,
			normalize_features = normalize_features,
			frontend = frontend
		)


class Wav2LetterDenseBigLargeKernelsNoDilationNoTemporalMaskNoDropoutReLu(JasperNet):
	def __init__(
		self,
		num_input_features,
		num_classes,
		dropout = 0.0,
		base_width = 128,
		nonlinearity = ('relu', ),
		kernel_size_prologue = 11,
		kernel_size_epilogue = 29,
		kernel_sizes = [11, 13, 17, 21, 25],
		dilation = 1,
		num_blocks = 5,
		decoder_type = None,
		normalize_features = True,
		frontend = None
	):
		super().__init__(
			num_input_features,
			num_classes,
			base_width = base_width,
			num_subblocks = 2,
			dropout = dropout,
			dropout_prologue = dropout,
			dropout_epilogue = dropout,
			dropouts = [dropout] * num_blocks,
			kernel_size_prologue = kernel_size_prologue,
			kernel_size_epilogue = kernel_size_epilogue,
			kernel_sizes = kernel_sizes,
			out_width_factors = [2, 3, 4, 5, 6],
			out_width_factors_large = [7, 8],
			residual = 'dense',
			dilation = dilation,
			nonlinearity = nonlinearity,
			decoder_type = decoder_type,
			normalize_features = normalize_features,
			frontend = frontend,
			temporal_mask = False
		)


class Wav2LetterFlat(JasperNet):
	def __init__(
		self,
		num_input_features,
		num_classes,
		dropout = 0.2,
		base_width = 128,
		nonlinearity = ('hardtanh', 0, 20),
		kernel_size_prologue = 13,
		kernel_size_epilogue = 29,
		kernel_sizes = [11, 13, 17, 21, 25],
		dilation = 2,
		num_blocks = 5,
		decoder_type = None,
		normalize_features = True,
		frontend = None
	):
		super().__init__(
			num_input_features,
			num_classes,
			base_width = base_width,
			dropout = dropout,
			dropout_prologue = dropout,
			dropout_epilogue = dropout,
			dropouts = [dropout] * num_blocks,
			kernel_size_prologue = kernel_size_prologue,
			kernel_size_epilogue = kernel_size_epilogue,
			kernel_sizes = [kernel_size_prologue] * num_blocks,
			out_width_factors = [6] * num_blocks,
			out_width_factors_large = [16, 16],
			residual = 'flat',
			dilation = dilation,
			nonlinearity = nonlinearity,
			decoder_type = decoder_type,
			normalize_features = normalize_features,
			frontend = frontend
		)


class JasperNetSeparable(JasperNet):
	def __init__(self, *args, separable = True, groups = 128, **kwargs):
		super().__init__(*args, separable = separable, groups = groups, **kwargs)


class JasperNetSmall(JasperNet):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, num_subblocks = 1, temporal_mask = False, **kwargs)


class JasperNetSmallInstanceNorm(JasperNet):
	def __init__(self, *args, **kwargs):
		super().__init__(
			*args,
			num_subblocks = 1,
			temporal_mask = False,
			normalize_features_legacy = False,
			normalize_features_temporal_mask = False,
			**kwargs
		)


class JasperNetSmallTrainableInstanceNorm(JasperNet):
	def __init__(self, *args, **kwargs):
		super().__init__(
			*args,
			num_subblocks = 1,
			temporal_mask = False,
			normalize_features_legacy = False,
			normalize_features_track_running_stats = True,
			normalize_features_temporal_mask = False,
			**kwargs
		)


class JasperNetBig(JasperNet):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, num_subblocks = 2, temporal_mask = False, **kwargs)


class JasperNetBigNoStride(JasperNet):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, num_subblocks = 2, stride1 = 1, temporal_mask = False, **kwargs)


class JasperNetBigBpeOnly(JasperNet):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, num_subblocks = 2, temporal_mask = False, bpe_only = True, **kwargs)


class JasperNetResidualBig(JasperNet):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, num_subblocks = 2, temporal_mask = False, residual = True, **kwargs)


class JasperNetBigInplace(JasperNet):
	def __init__(self, *args, **kwargs):
		inplace = kwargs.pop('inplace', True)
		super().__init__(
			*args,
			num_subblocks = 2,
			temporal_mask = False,
			inplace = inplace,
			nonlinearity = ('leaky_relu', 0.01),
			**kwargs
		)
