'''
Traceback (most recent call last):
File "train.py", line 1078, in
main(args)
File "train.py", line 472, in main
torch.onnx.export(
File "/opt/conda/lib/python3.8/site-packages/torch/onnx/init.py", line 271, in export
return utils.export(model, args, f, export_params, verbose, training,
File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 88, in export
_export(model, args, f, export_params, verbose, training, input_names, output_names,
File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 691, in _export
_model_to_graph(model, args, verbose, input_names,
File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 460, in _model_to_graph
graph = _optimize_graph(graph, operator_export_type,
File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 206, in _optimize_graph
graph = torch._C._jit_pass_onnx(graph, operator_export_type)
File "/opt/conda/lib/python3.8/site-packages/torch/onnx/init.py", line 309, in _run_symbolic_function
return utils._run_symbolic_function(*args, **kwargs)
File "/opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py", line 994, in _run_symbolic_function
return symbolic_fn(g, *inputs, **attrs)
File "/opt/conda/lib/python3.8/site-packages/torch/onnx/symbolic_helper.py", line 148, in wrapper
return fn(g, *args, **kwargs)
File "/opt/conda/lib/python3.8/site-packages/torch/onnx/symbolic_opset13.py", line 73, in split
size = self.type().sizes()[dim]
TypeError: 'NoneType' object is not subscriptable
(Occurred when translating split).
'''