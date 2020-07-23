import gc
import sys
import traceback
import torch


def handle_out_of_memory_exception(model_parameters = []):
	exc_type, exc_value, exc_traceback = sys.exc_info()
	if 'out of memory' in str(exc_value):
		print('RECOVERING FROM OOM --- BEFORE FREE')
		traceback.print_exception(exc_type, exc_value, exc_traceback)
		for p in model_parameters:
			p.grad = None
		print_memory_stats('<BEFORE>')
		free_up_memory()
		print_memory_stats('<AFTER>')
		print('RECOVERING FROM OOM --- AFTER FREE')
		return True
	return False


def free_up_memory(reset_counters = False):
	gc.collect()
	if torch.cuda.is_available:
		torch.cuda.empty_cache()
	gc.collect()
	if torch.cuda.is_available and reset_counters:
		torch.cuda.reset_peak_memory_stats()


def print_memory_stats(prefix = '', scaler = dict(mb = 1e6)):
	k, v = next(iter(scaler.items()))
	for device in range(torch.cuda.is_available and torch.cuda.device_count() or 0):
		print(
			'MEMORY',
			prefix,
			'reserved',
			torch.cuda.memory_reserved(device) / v,
			'allocated',
			torch.cuda.memory_allocated(device) / v,
			k
		)
		print(
			'MEMORY MAX',
			prefix,
			'reserved',
			torch.cuda.max_memory_reserved(device) / v,
			'allocated',
			torch.cuda.max_memory_allocated(device) / v,
			k
		)


def enable_jit_fusion():
	torch._C._jit_set_profiling_executor(False)
	torch._C._jit_set_profiling_mode(False)
	torch._C._jit_override_can_fuse_on_gpu(True)
	torch._C._jit_set_texpr_fuser_enabled(False)
