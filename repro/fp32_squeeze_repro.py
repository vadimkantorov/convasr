import onnxruntime
from torch import nn
import torch
import torch.nn.functional as F

print(torch.__version__)


class Model(nn.Module):
	def forward(self, x):
		p = F.pad(x, (1, 1), mode='constant').squeeze(1)
		return dict(o=p)


model = Model()
model.to(device='cuda')
x = torch.rand(1, 10).to(device='cuda')
print(x.shape)
print(model(x))

torch.onnx.export(
		model, (x,),
		'model.onnx',
		verbose=False,
		opset_version=12,
		export_params=None,
		do_constant_folding=True,
		input_names=['x'],
		output_names = ['o']
)

runtime = onnxruntime.InferenceSession('model.onnx')
print(runtime.run(None, dict(x=x.cpu().numpy())))

'''
1.7.1
torch.Size([1, 10])
{'o': tensor([[0.0000, 0.0726, 0.5379, 0.3110, 0.6998, 0.3462, 0.9140, 0.1063, 0.3140,
         0.2962, 0.8441, 0.0000]], device='cuda:0')}
2021-03-19 22:33:08.805984358 [W:onnxruntime:Default, fallback_cpu_capability.h:140 GetCpuPreferedNodes] Force fallback to CPU execution for node: Gather_17
2021-03-19 22:33:08.806113942 [W:onnxruntime:Default, fallback_cpu_capability.h:140 GetCpuPreferedNodes] Force fallback to CPU execution for node: Equal_19
Traceback (most recent call last):
  File "/home/yuborovskikh/work/convasr/repro/fp32_squeeze_repro.py", line 33, in <module>
    print(runtime.run(None, dict(x=x.cpu().numpy())))
  File "/opt/conda/lib/python3.8/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 120, in run
    raise ValueError("Model requires {} inputs. Input Feed contains {}".format(num_required_inputs, num_inputs))
ValueError: Model requires 3 inputs. Input Feed contains 1

Process finished with exit code 1
'''