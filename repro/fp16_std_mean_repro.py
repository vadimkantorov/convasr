import onnxruntime
from torch import nn
import torch

print(torch.__version__)


class Model(nn.Module):
	def forward(self, x):
		std, mean = torch.std_mean(x)
		return dict(o=std)


model = Model()
model.to(torch.float16)
x = torch.rand(10).to(torch.float16)

print(model(x))

torch.onnx.export(
		model, (x,),
		'fp16_std_mean_repro.onnx',
		verbose=False,
		opset_version=12,
		export_params=None,
		do_constant_folding=True,
		input_names=['x']
)

runtime = onnxruntime.InferenceSession('fp16_std_mean_repro.onnx')
print(runtime.run(None, dict(x=x.cpu().numpy())))

'''
1.7.1
{'o': tensor(0.2499, dtype=torch.float16)}
Traceback (most recent call last):
  File "/home/*/work/convasr/repro/fp16_std_mean_repro.py", line 19, in <module>
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
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 930, in _run_symbolic_function
    symbolic_fn = _find_symbolic_in_registry(domain, op_name, opset_version, operator_export_type)
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 888, in _find_symbolic_in_registry
    return sym_registry.get_registered_op(op_name, domain, opset_version)
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/symbolic_registry.py", line 111, in get_registered_op
    raise RuntimeError(msg)
RuntimeError: Exporting the operator std_mean to ONNX opset version 12 is not supported. Please open a bug to request ONNX export support for the missing operator.

Process finished with exit code 1

1.8.0
{'o': tensor(0.2969, dtype=torch.float16)}
'''