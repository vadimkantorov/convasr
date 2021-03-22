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
		'test_output.onnx',
		verbose=False,
		opset_version=12,
		export_params=None,
		do_constant_folding=True,
		input_names=['x', 'xlen']
)

runtime = onnxruntime.InferenceSession('test_output.onnx')
print(runtime.run(None, dict(x=x.cpu().numpy(), xlen=xlen.cpu().numpy())))

'''
1.8.0
{'o': tensor([[0.7016, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.3032, 0.2104, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000]], device='cuda:0')}
Traceback (most recent call last):
  File "fp32_jit_prim_dtype_repro.py", line 41, in <module>
    runtime = onnxruntime.InferenceSession('test_output.onnx')
  File "/opt/conda/lib/python3.8/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 206, in __init__
    self._create_inference_session(providers, provider_options)
  File "/opt/conda/lib/python3.8/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 231, in _create_inference_session
    sess.initialize_session(providers or [], provider_options or [])
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Node (Mul_48) Op (Mul) [ShapeInferenceError] Incompatible dimensions
'''
