import onnxruntime
from torch import nn
import torch

print(torch.__version__)

@torch.jit.script
def compute_output_lengths(x, lengths_fraction):
	if lengths_fraction is None:
		return torch.full(x.shape[:1], x.shape[-1], device = x.device, dtype = torch.long)
	return (lengths_fraction * x.shape[-1]).ceil().long()

class StdMeanForExport(nn.Module):
	def forward(self, x, l):
		l = compute_output_lengths(x, l)
		return dict(o=l)

model = StdMeanForExport()
model.to(device='cuda')

x = torch.rand(1, 10).to(device='cuda')
l = torch.rand(1).to(device='cuda')

print(model(x, l))

torch.onnx.export(
		model, (x, l),
		'test_output.onnx',
		verbose=False,
		opset_version=12,
		export_params=None,
		do_constant_folding=True,
		input_names=['x', 'l']
)

runtime = onnxruntime.InferenceSession('test_output.onnx')

'''
1.7.1
{'o': tensor([8, 4, 5, 2, 7, 7, 7, 5, 9, 7], device='cuda:0')}
Warning: ONNX Scalar Type Analysis - Scalar types mismatch for tensor inputs of operator onnx::Mul. Please report a bug to PyTorch. The scalar type Float of the first tensor is chosen.

Process finished with exit code 0
'''
