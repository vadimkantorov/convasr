import onnxruntime
from torch import nn
import torch

print(torch.__version__)

@torch.jit.script
def compute_output_lengths(x, lengths_fraction):
	if lengths_fraction is None:
		return torch.full(x.shape[:1], x.shape[-1], device = x.device, dtype = torch.long)
	return (lengths_fraction * x.shape[-1]).ceil().long()

@torch.jit.script
def temporal_mask(x, lengths):
	return (torch.arange(x.shape[-1], device = x.device, dtype = torch.long).unsqueeze(0) <
			lengths.unsqueeze(1)).view(x.shape[:1] + (1, ) * (len(x.shape) - 2) + x.shape[-1:])

class StdMeanForExport(nn.Module):
	def forward(self, x, xlen):
		l = compute_output_lengths(x, xlen)
		tm = temporal_mask(x, l)
		x = x * tm
		return dict(o=x)

model = StdMeanForExport()
model.to(dtype=torch.float16, device='cuda')

x = torch.rand(2, 10).to(dtype=torch.float16, device='cuda')
xlen = torch.rand(2).to(dtype=torch.float16, device='cuda')

print(model(x, xlen))

torch.onnx.export(
		model, (x, xlen),
		'test_output.onnx',
		verbose=False,
		opset_version=12,
		export_params=None,
		do_constant_folding=True,
		input_names=['x', 'xlen']
)

runtime = onnxruntime.InferenceSession('test_output.onnx')
print(runtime.run(None, dict(x=x.cpu().numpy(), xlen=xlen.cpy().numpy())))

'''
1.7.1
{'o': tensor([[0.0296, 0.5049, 0.7402, 0.4216, 0.7007, 0.7080, 0.0146, 0.7407, 0.0000,
         0.0000],
        [0.2290, 0.6479, 0.3743, 0.8706, 0.7949, 0.0840, 0.5420, 0.4280, 0.0000,
         0.0000]], device='cuda:0', dtype=torch.float16)}
Warning: ONNX Scalar Type Analysis - Scalar types mismatch for tensor inputs of operator onnx::Mul. Please report a bug to PyTorch. The scalar type Half of the first tensor is chosen.

Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
'''
