import apex
import torch
from torch import nn


class OneConvModel(nn.Module):
	def __init__(self, kernel_size = 11,
			stride = 2,
			dilation = 1,
			groups = 1,
			in_channels = 64,
			base_wide = 128,
			out_width_factors = [2],
			*args, **kwargs):
		super().__init__()

		in_width_factor = out_width_factors[0]
		padding = dilation * kernel_size // 2
		out_channels = base_wide * in_width_factor
		self.conv = nn.ModuleList([
			nn.Conv1d(
					in_channels,
					out_channels,
					kernel_size=kernel_size,
					stride=stride,
					padding=padding,
					dilation=dilation,
					groups=groups
			)
		])

	def forward(self, x):  # x: typing.Union[shaping.BCT]
		assert len(x.shape) == 3

		for i, subblock in enumerate(self.conv):
			x = subblock(x)

		return dict(out=x)

	def fuse_conv_bn_eval(self, K = None):
		pass


def data_parallel_and_autocast(model, optimizer = None, data_parallel = True, opt_level = None, **kwargs):
	data_parallel = data_parallel and torch.cuda.device_count() > 1
	model_training = model.training

	if opt_level is None:
		model = torch.nn.DataParallel(model) if data_parallel else model

	elif data_parallel:
		model, optimizer = apex.amp.initialize(nn.Sequential(model), optimizers = optimizer, opt_level = opt_level, **kwargs) if optimizer is not None else (apex.amp.initialize(nn.Sequential(model), opt_level = opt_level, **kwargs), None)
		model = torch.nn.DataParallel(model[0])
		model.forward = lambda *args, old_fwd = model.forward, input_caster = lambda tensor: tensor.to(apex.amp._amp_state.opt_properties.options['cast_model_type']) if tensor.is_floating_point() else tensor, output_caster = lambda tensor: (tensor.to(apex.amp._amp_state.opt_properties.options['cast_model_outputs'] if apex.amp._amp_state.opt_properties.options.get('cast_model_outputs') is not None else torch.float32)) if tensor.is_floating_point() else tensor, **kwargs: apex.amp._initialize.applier(old_fwd(*apex.amp._initialize.applier(args, input_caster), **apex.amp._initialize.applier(kwargs, input_caster)), output_caster)

	else:
		model, optimizer = apex.amp.initialize(model, optimizers = optimizer, opt_level = opt_level, **kwargs) if optimizer is not None else (apex.amp.initialize(model, opt_level = opt_level, **kwargs), None)

	model.train(model_training)
	return model, optimizer
