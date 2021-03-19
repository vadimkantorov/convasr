import onnxruntime
from torch import nn
import torch

print(torch.__version__)


class StdMeanForExport(nn.Module):
	def forward(self, x):
		x = x.pow(2)
		return dict(o=x)


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
		input_names=['x']
)

runtime = onnxruntime.InferenceSession('test_output.onnx')
print(runtime.run(None, dict(x=x.cpu().numpy())))

'''
1.7.1
{'o': tensor([5.9424e-01, 6.5625e-01, 5.1367e-01, 6.6109e-03, 2.3425e-04, 5.9229e-01,
        2.1851e-02, 7.5293e-01, 6.6895e-01, 1.4612e-01], device='cuda:0',
       dtype=torch.float16)}
2021-03-19 21:25:54.311085679 [W:onnxruntime:Default, cuda_execution_provider.cc:1885 GetCapability] CUDA kernel not found in registries for Op type: Pow node name: Pow_1
[array([5.942e-01, 6.562e-01, 5.137e-01, 6.611e-03, 2.342e-04, 5.923e-01,
       2.185e-02, 7.529e-01, 6.689e-01, 1.461e-01], dtype=float16)]

Process finished with exit code 0
'''