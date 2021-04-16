import onnxruntime
from torch import nn
import torch
import torch.nn.functional as F

print(torch.__version__)


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.sftt = nn.Conv1d(1, 258, 256, bias=False, stride=80)

	def forward(self, x):
		pad = 5
		padded_signal = F.pad(x, (0, pad), mode='constant', value=0)
		stft_res = self.sftt(padded_signal.unsqueeze(dim=1))
		real_squared, imag_squared = (stft_res * stft_res).split(129, dim=1)
		return dict(o=real_squared)


model = Model()
x = torch.rand((2, 1024))
print(x.shape)

torch.onnx.export(
		model, (x,),
		'fp16_opset_13.onnx',
		verbose=False,
		opset_version=13,
		export_params=True,
		do_constant_folding=True,
		input_names=['x']
)

runtime = onnxruntime.InferenceSession('fp16_opset_13.onnx')
print(runtime.run(None, dict(x=x.cpu().numpy())))

'''
1.8.1
torch.Size([2, 1024])
Traceback (most recent call last):
  File "repro/fp32_opset_13.py", line 26, in <module>
    torch.onnx.export(
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/__init__.py", line 271, in export
    return utils.export(model, args, f, export_params, verbose, training,
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 88, in export
    _export(model, args, f, export_params, verbose, training, input_names, output_names,
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 694, in _export
    _model_to_graph(model, args, verbose, input_names,
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 463, in _model_to_graph
    graph = _optimize_graph(graph, operator_export_type,
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 206, in _optimize_graph
    graph = torch._C._jit_pass_onnx(graph, operator_export_type)
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/__init__.py", line 309, in _run_symbolic_function
    return utils._run_symbolic_function(*args, **kwargs)
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 997, in _run_symbolic_function
    return symbolic_fn(g, *inputs, **attrs)
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/symbolic_helper.py", line 148, in wrapper
    return fn(g, *args, **kwargs)
  File "/opt/conda/lib/python3.8/site-packages/torch/onnx/symbolic_opset13.py", line 73, in split
    size = self.type().sizes()[dim]
TypeError: 'NoneType' object is not subscriptable 
(Occurred when translating split).

'''
