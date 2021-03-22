import onnxruntime
from torch import nn
import torch

print(torch.__version__)


class Model(nn.Module):
	def forward(self, x):
		x = x.pow(2)
		return dict(o=x)


model = Model()
model.to(dtype=torch.float16, device='cuda')
x = torch.rand(10).to(dtype=torch.float16, device='cuda')

print(model(x))

torch.onnx.export(
		model, (x,),
		'fp16_pow_repro.onnx',
		verbose=False,
		opset_version=12,
		export_params=None,
		do_constant_folding=True,
		input_names=['x']
)

runtime = onnxruntime.InferenceSession('fp16_pow_repro.onnx')
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

1.8.0
{'o': tensor([2.1899e-01, 9.2920e-01, 4.8280e-05, 7.2607e-01, 1.1436e-02, 7.0605e-01,
        4.9622e-02, 1.7566e-01, 6.3867e-01, 4.8755e-01], device='cuda:0',
       dtype=torch.float16)}
2021-03-22 14:44:36.656817045 [W:onnxruntime:Default, cuda_execution_provider.cc:1885 GetCapability] CUDA kernel not found in registries for Op type: Pow node name: Pow_1
[array([2.190e-01, 9.292e-01, 4.828e-05, 7.261e-01, 1.144e-02, 7.061e-01,
       4.962e-02, 1.757e-01, 6.387e-01, 4.875e-01], dtype=float16)]

'''