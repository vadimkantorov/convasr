import onnxruntime
from torch import nn
import torch

print(torch.__version__)


class StdMeanForExport(nn.Module):
	def forward(self, x):
		std = torch.std(x)
		return dict(o=std)

model = StdMeanForExport()
model.to(dtype=torch.float16, device='cuda')

x = torch.rand(10).to(dtype=torch.float16, device='cuda')

print(model(x))

torch.onnx.export(
		model, (x,),
		'test_output.onnx',
		verbose=False,
		opset_version=12,
		export_params=None,
		do_constant_folding=True,
		input_names=['x'],
		output_names=['o'],
		dynamic_axes=dict(x={
			0: 'B',
		}, o={
			0: 'B',
		})
)

runtime = onnxruntime.InferenceSession('test_output.onnx')
print(runtime.run(None, dict(x=x.cpu().numpy())))

'''
1.7.1
{'o': tensor(0.2084, device='cuda:0', dtype=torch.float16)}
Traceback (most recent call last):
  File "/home/*/work/convasr/repro/fp16_std_repro_cuda.py", line 36, in <module>
    runtime = onnxruntime.InferenceSession('test_output.onnx')
  File "/opt/conda/lib/python3.8/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 206, in __init__
    self._create_inference_session(providers, provider_options)
  File "/opt/conda/lib/python3.8/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 226, in _create_inference_session
    sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Load model from test_output.onnx failed:Type Error: Type parameter (T) bound to different types (tensor(float16) and tensor(float) in node (Mul_7).
'''
