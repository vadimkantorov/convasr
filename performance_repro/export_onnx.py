import onnxruntime
import torch

from models import OneConvModel

model = OneConvModel()

torch.set_grad_enabled(False)
model.eval()
model.to('cuda')
model.fuse_conv_bn_eval()

dtype = torch.float16

model = model.to(dtype)
waveform_input = torch.rand((4, 64, 128), device='cuda', dtype=dtype)

logits = model(waveform_input)
print(logits)

torch.onnx.export(
		model, (waveform_input),
		'conv_fp16.onnx',
		opset_version=12,
		export_params=True,
		do_constant_folding=True,
		input_names=['x'],
		output_names=['out'],
		dynamic_axes=dict(x={
			0: 'B',
			2: 'T'
		}, out={
			0: 'B',
			2: 't'
		})
)

onnxruntime_session = onnxruntime.InferenceSession('conv_fp16.onnx')
(logits_,) = onnxruntime_session.run(None, dict(x=waveform_input.cpu().to(dtype=dtype).numpy()))

assert torch.allclose(
		logits['out'].cpu(),
		torch.from_numpy(logits_),
		**{
			'rtol': 1e-01,
			'atol': 1e-02
		}
)
