from torch import nn
import torch

print(torch.__version__)


class StdMeanForExport(nn.Module):
	def forward(self, x):
		std = torch.std(x)
		return dict(o=std)

model = StdMeanForExport()
model.to(dtype=torch.float16)

x = torch.rand(10).to(dtype=torch.float16)

print(model(x))


'''
1.7.1
Traceback (most recent call last):
  File "/home/*/work/convasr/repro/fp16_std_repro_cpu.py", line 18, in <module>
    print(model(x))
  File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/*/work/convasr/repro/fp16_std_repro_cpu.py", line 10, in forward
    std = torch.std(x)
RuntimeError: _th_std not supported on CPUType for Half
'''