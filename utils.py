import gc
import sys
import traceback
import random
import torch
import gzip
import psutil
import logging
import logging.handlers

def get_root_logger_print():
	logger = logging.getLogger()
	return (lambda *args: logger.info(' '.join(map(str, args))))

def set_up_root_logger(log_file_path = None, mode = 'a', max_bytes = 1_000_000, fmt = '%(asctime)s [%(levelname)s]: %(message)s'):
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	
	formatter = logging.Formatter(fmt)
	handler = logging.StreamHandler()
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	
	if log_file_path:
		handler = logging.handlers.RotatingFileHandler(log_file_path, maxBytes = max_bytes, backupCount = 0)
		# workaround logging logic to enforce simple file append/create logic
		handler.stream.close()
		handler.stream = open(log_file_path, mode)
		handler.setFormatter(formatter)
		logger.addHandler(handler)

def open_maybe_gz(data_path):
	return gzip.open(data_path, 'rt') if data_path.endswith('.gz') else open(data_path)

def compute_memory_stats(byte_scaler = 1024**3, measure_pss_ram = False):
	device_count = torch.cuda.device_count()
	total_allocated = 0
	total_reserved = 0

	res = {}
	for i in range(device_count):
		device_stats = torch.cuda.memory_stats(i)
		allocated = device_stats['allocated_bytes.all.peak'] / byte_scaler
		total_allocated += allocated

		reserved = device_stats[f'reserved_bytes.all.peak'] / byte_scaler
		total_reserved += reserved
		res[f'allocated_cuda{i}'] = allocated
		res[f'reserved_cuda{i}'] = reserved

	res['allocated'] = total_allocated
	res['reserved'] = total_reserved 

	if measure_pss_ram:
		process = psutil.Process()
		children = process.children(recursive=True)
		total_pss_ram = process.memory_full_info().pss + sum(
			child.memory_full_info().pss for child in children
		)
		res['pss_ram'] = total_pss_ram / byte_scaler
	else:
		res['pss_ram'] = 0.0
	return res

def reset_cpu_threads(num_threads):
	torch.set_num_threads(num_threads)
	#os.environ['OMP_NUM_THREADS'] = str(num_threads)
	#os.environ['MKL_NUM_THREADS'] = str(num_threads)

def set_random_seed(seed):
	for set_random_seed in [random.seed, torch.manual_seed
							] + ([torch.cuda.manual_seed_all] if torch.cuda.is_available() else []):
		set_random_seed(seed)

class OomHandler:
	def __init__(self, max_retries = 0):
		self.retries = 0
		self.max_retries = max_retries

	def reset(self):
		self.retries = 0

	def try_recover(self, model_parameters = [], _print = print):
		exc_type, exc_value, exc_traceback = sys.exc_info()
		if 'out of memory' in str(exc_value):
			self.retries += 1
			if self.retries > self.max_retries:
				return False
			
			_print('RECOVERING FROM OOM --- BEFORE FREE')
			traceback.print_exception(exc_type, exc_value, exc_traceback)
			for p in model_parameters:
				p.grad = None
			print_memory_stats('<BEFORE FREE>', _print = _print)
			free_up_memory()
			print_memory_stats('<AFTER FREE>', _print = _print)
			_print('RECOVERING FROM OOM --- AFTER FREE', _print = _print)
			return True
		return False


def free_up_memory(reset_counters = False):
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		if reset_counters:
			torch.cuda.reset_peak_memory_stats()
	gc.collect()


def print_memory_stats(prefix = '', scaler = dict(mb = 1e6), _print = print):
	k, v = next(iter(scaler.items()))
	for device in range(torch.cuda.device_count()):
		_print(
			'MEMORY',
			prefix,
			'reserved',
			torch.cuda.memory_reserved(device) / v,
			'allocated',
			torch.cuda.memory_allocated(device) / v,
			k
		)
		_print(
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
