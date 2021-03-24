import onnxruntime
from torch import nn
import torch

print(torch.__version__)

@torch.jit.script
def compute_output_lengths(x, lengths_fraction):
	return (lengths_fraction * x.shape[-1]).ceil().long()

@torch.jit.script
def temporal_mask(x, lengths):
	return (torch.arange(x.shape[-1], device = x.device, dtype = lengths.dtype).unsqueeze(0) <
			lengths.unsqueeze(1)).view(x.shape[:1] + (1, ) * (len(x.shape) - 2) + x.shape[-1:])

class Model(nn.Module):
	def forward(self, x, xlen):
		l = compute_output_lengths(x, xlen)
		tm = temporal_mask(x, l)
		x = x * tm
		return dict(o=x)

model = Model()
model.to(device='cuda')

x = torch.rand(2, 10).to( device='cuda')
xlen = torch.rand(2).to( device='cuda')

print(model(x, xlen))

torch.onnx.export(
		model, (x, xlen),
		'fp32_jit_prim_dtype_repro.onnx',
		verbose=False,
		opset_version=12,
		export_params=None,
		do_constant_folding=True,
		input_names=['x', 'xlen']
)

runtime = onnxruntime.InferenceSession('fp32_jit_prim_dtype_repro.onnx')
print(runtime.run(None, dict(x=x.cpu().numpy(), xlen=xlen.cpu().numpy())))

'''
1.7.1
{'o': tensor([[0.9552, 0.6199, 0.5749, 0.0528, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.2690, 0.3177, 0.9293, 0.6636, 0.9397, 0.9952, 0.6596, 0.2583, 0.0496,
         0.8314]], device='cuda:0')}
Traceback (most recent call last):
  File "fp32_jit_prim_dtype_repro.py", line 31, in <module>
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


1.8.0
{'o': tensor([[0.7016, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.3032, 0.2104, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000]], device='cuda:0')}
Traceback (most recent call last):
  File "fp32_jit_prim_dtype_repro.py", line 41, in <module>
    runtime = onnxruntime.InferenceSession('fp32_jit_prim_dtype_repro.onnx')
  File "/opt/conda/lib/python3.8/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 206, in __init__
    self._create_inference_session(providers, provider_options)
  File "/opt/conda/lib/python3.8/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 231, in _create_inference_session
    sess.initialize_session(providers or [], provider_options or [])
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Node (Mul_48) Op (Mul) [ShapeInferenceError] Incompatible dimensions
'''
