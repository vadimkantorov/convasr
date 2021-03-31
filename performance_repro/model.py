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
