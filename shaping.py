import functools
import typing
import torch

# equal to 1T
class _T(torch.Tensor):
	pass

class BY(torch.Tensor):
	pass

class T(torch.Tensor):
	pass

class B(torch.Tensor):
	pass

class S(torch.Tensor):
	pass

class BCT(torch.Tensor):
	pass

class CT(torch.Tensor):
	pass

class BCt(torch.Tensor):
	pass

class Bt(torch.Tensor):
	pass

class TBC(torch.Tensor):
	pass

class BT(torch.Tensor):
	pass

class BLY(torch.Tensor):
	pass

class BS(torch.Tensor):
	pass

def is_tensor_hint(cls):
	return issubclass(cls, torch.Tensor)

def unbind_tensor_hint(cls):
	dims = cls.__name__.split('.')[-1]
	return dims

def shapecheck(hints = None, auto = None, **kwargs):
	if auto is not None:
		def decorator(fn):
			@functools.wraps(fn)
			def wrapper(*args, **kwargs):
				shapecheck.hints = typing.get_type_hints(fn)
				if auto:
					shapecheck(hints = {}, **kwargs)
				res = fn(*args, **kwargs)
				if auto:
					shapecheck(hints = {}, **kwargs, **{'return' : res})
				shapecheck.hints = {}
				return res
			return wrapper
		return decorator

	else:
		hints = hints or shapecheck.hints
		dims = {}
		for k, v in kwargs.items():
			h = hints.get(k)
			if h is not None:
				if is_tensor_hint(h):
					tensor_dims = unbind_tensor_hint(h)
					assert v.ndim == len(tensor_dims), f'Tensor [{k}] should be typed [{tensor_dims}] and should have rank {len(tensor_dims)} but has rank [v.ndim]'
					for i, d in enumerate(tensor_dims):
						s = v.shape[i]
						if d in dims:
							assert dims[d] == s, f'Tensor [{k}] should be typed [{tensor_dims}], dim [{d}] should have rank [{dims[d]}] but has rank [{s}]'
						dims[d] = s
				else:
					assert isinstance(v, h), f'Arg [{k}] should be typed [{h}] but is typed [{type(v)}]'
					
