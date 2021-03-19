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
	return (torch.arange(x.shape[-1], device = x.device, dtype = lengths.dtype).unsqueeze(0) <
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
{'o': tensor([[0.1605, 0.5532, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.1528, 0.4634, 0.4460, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000]], device='cuda:0', dtype=torch.float16)}
Traceback (most recent call last):
  File "/home/yuborovskikh/work/convasr/repro/fp16_jit_prim_dtype_repro.py", line 33, in <module>
    torch.onnx.export(
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/__init__.py", line 225, in export
    return utils.export(model, args, f, export_params, verbose, training,
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 85, in export
    _export(model, args, f, export_params, verbose, training, input_names, output_names,
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 632, in _export
    _model_to_graph(model, args, verbose, input_names,
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 417, in _model_to_graph
    graph = _optimize_graph(graph, operator_export_type,
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 203, in _optimize_graph
    graph = torch._C._jit_pass_onnx(graph, operator_export_type)
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/__init__.py", line 263, in _run_symbolic_function
    return utils._run_symbolic_function(*args, **kwargs)
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 978, in _run_symbolic_function
    symbolic_fn = _find_symbolic_in_registry(domain, symbolic_name, opset_version,
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 888, in _find_symbolic_in_registry
    return sym_registry.get_registered_op(op_name, domain, opset_version)
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/symbolic_registry.py", line 111, in get_registered_op
    raise RuntimeError(msg)
RuntimeError: Exporting the operator prim_dtype to ONNX opset version 12 is not supported. Please open a bug to request ONNX export support for the missing operator.

Process finished with exit code 1
'''
