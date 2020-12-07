import os
import shutil
import gc
import sys
import traceback
import random
import functools
import gzip
import numpy
import logging
import typing
import logging.handlers
import torch.distributed
import psutil
import torch

def flatten(lists):
	return functools.reduce(lambda acc, l: acc.extend(l) or acc, lists, [])

def strip_suffixes(s, suffixes):
	for suffix in sorted(suffixes, key = len, reverse = True):
		if s.endswith(suffix):
			return s[:-len(suffix)]
	return s

def get_root_logger_print(level = logging.INFO):
	logger = logging.getLogger()
	return (lambda *args: logger.log(level, ' '.join(map(str, args))))


def set_up_root_logger(log_file_path = None, mode = 'a', max_bytes = 1_000_000, fmt = '%(asctime)s [%(levelname)s]: %(message)s', level = logging.INFO):
	logger = logging.getLogger()
	logger.setLevel(level)

	formatter = logging.Formatter(fmt)
	handler = logging.StreamHandler()
	handler.setFormatter(formatter)
	# hack to avoid duplicate messages in stdout
	handler.addFilter(lambda record: record.levelno != logging.CRITICAL)
	logger.addHandler(handler)

	if log_file_path:
		handler = logging.handlers.RotatingFileHandler(log_file_path, maxBytes = max_bytes, backupCount = 0)
		# workaround logging logic to enforce simple file append/create logic
		handler.stream.close()
		handler.stream = open(log_file_path, mode)
		handler.setFormatter(formatter)
		logger.addHandler(handler)


def open_maybe_gz(data_path, mode = 'r'):
	return gzip.open(data_path, mode + 't') if data_path.endswith('.gz') else open(data_path, mode)


def compute_cuda_memory_stats(byte_scaler = 1024**3, devices = None):
	if devices is None:
		devices = range(torch.cuda.device_count())
	total_allocated = 0
	total_reserved = 0

	stats = {}
	for i in devices:
		device_stats = torch.cuda.memory_stats(i)
		allocated = device_stats['allocated_bytes.all.peak'] / byte_scaler
		total_allocated += allocated

		reserved = device_stats[f'reserved_bytes.all.peak'] / byte_scaler
		total_reserved += reserved
		stats[f'allocated_cuda{i}'] = allocated
		stats[f'reserved_cuda{i}'] = reserved

	stats['allocated'] = total_allocated
	stats['reserved'] = total_reserved
	return stats


def compute_ram_memory_stats(byte_scaler =1024 ** 3):
	stats = {}
	process = psutil.Process()
	children = process.children(recursive=True)
	total_pss_ram = process.memory_full_info().pss + sum(
		child.memory_full_info().pss for child in children
	)
	stats['pss_ram'] = total_pss_ram / byte_scaler
	return stats


def compute_memory_fragmentation():
	snapshot = torch.cuda.memory_snapshot()
	return sum(b['allocated_size'] for b in snapshot) / sum(b['total_size'] for b in snapshot)

def open_maybe_gz(data_path, mode = 'r'):
	return gzip.open(data_path, mode + 't') if data_path.endswith('.gz') else open(data_path, mode)

def reset_cpu_threads(num_threads):
	torch.set_num_threads(num_threads)
	#os.environ['OMP_NUM_THREADS'] = str(num_threads)
	#os.environ['MKL_NUM_THREADS'] = str(num_threads)

def set_random_seed(seed):
	for set_random_seed in [random.seed, torch.manual_seed, numpy.random.seed
							] + ([torch.cuda.manual_seed_all] if torch.cuda.is_available() else []):
		set_random_seed(seed)


def copy_tensorboard_dir_from_previous_checkpoint_if_exists(args, tensorboard_dir):
	if len(args.checkpoint) > 0 and args.experiment_name and args.local_rank == 0:
		tensorboard_dir_checkpoint = os.path.join(os.path.dirname(args.checkpoint[0]), 'tensorboard')
		if os.path.exists(tensorboard_dir_checkpoint) and not os.path.exists(tensorboard_dir):
			shutil.copytree(tensorboard_dir_checkpoint, tensorboard_dir)


class OomHandler:
	def __init__(self, max_retries = 0):
		self.retries = 0
		self.max_retries = max_retries

	def reset(self):
		self.retries = 0

	def try_recover(self, model_parameters = [], _print = print):
		exc_type, exc_value, exc_traceback = sys.exc_info()
		# TODO OOM restore doestn work in DDP setup, reason: https://github.com/pytorch/pytorch/issues/18853#issuecomment-698386652
		if 'out of memory' in str(exc_value):
			self.retries += 1
			if self.retries > self.max_retries:
				return False

			_print('RECOVERING FROM OOM --- BEFORE FREE')
			_print(''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
			for p in model_parameters:
				p.grad = None
			print_memory_stats('<BEFORE FREE>', _print = _print)
			free_up_memory()
			print_memory_stats('<AFTER FREE>', _print = _print)
			_print('RECOVERING FROM OOM --- AFTER FREE')
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


def gather_tensor_shapes(tensor: torch.Tensor, world_size: int) -> typing.List[torch.Tensor]:
	shape_tensor = torch.tensor(tensor.shape, dtype = torch.long, device = tensor.device)
	shapes = list(torch.zeros([world_size, len(tensor.shape)], dtype = torch.long, device = tensor.device).unbind(0))
	torch.distributed.all_gather(shapes, shape_tensor)
	return shapes


def gather_tensors(tensor: torch.Tensor, world_size: int) -> typing.List[torch.Tensor]:
	shapes = gather_tensor_shapes(tensor, world_size)
	max_shape = torch.stack([shape for shape in shapes], dim=0).max(dim=0).values
	padding = []
	for i, dim in enumerate(max_shape):
		padding += [0, dim.item() - tensor.size(i)]
	padded_tensor = torch.nn.functional.pad(tensor, padding)
	tensors = list(torch.zeros([world_size, *padded_tensor.shape], dtype=padded_tensor.dtype, device=padded_tensor.device).unbind(0))
	torch.distributed.all_gather(tensors, padded_tensor)
	for i, shape in enumerate(shapes):
		tensors[i] = tensors[i][list(map(lambda x: slice(x.item()), shape))]
	return tensors


class TensorBackedStringArray:
	def __init__(self, strings, encoding = 'utf_16_le', device = 'cpu'):
		strings = list(strings)
		self.encoding = encoding
		self.multiplier = dict(ascii = 1, utf_16_le = 2, utf_32_le = 4)[encoding]
		self.data = torch.ByteTensor(torch.ByteStorage.from_buffer(''.join(strings).encode(encoding))).to(device)
		self.cumlen = torch.tensor(list(map(len, strings)), dtype = torch.int64, device = device).cumsum(dim = 0)
		assert len(strings) == 0 or int(self.cumlen[-1]) * self.multiplier == len(self.data), f'[{encoding}] is not enough to hold characters, use a larger character class'

	def __getitem__(self, i):
		return bytes(self.data[(self.cumlen[i - 1] * self.multiplier if i >= 1 else 0) : self.cumlen[i] * self.multiplier]).decode(self.encoding)

	def __len__(self):
		return len(self.cumlen)

	def tolist(self):
		data = self.data.cpu().tolist()
		cumlen = [0] + (self.cumlen * self.multiplier).cpu().tolist()
		strings = []
		for start, end in zip(cumlen[:-1], cumlen[1:]):
			strings.append(bytes(data[start:end]).decode(self.encoding))
		return strings

	def to(self, device):
		self.data = self.data.to(device)
		self.cumlen = self.cumlen.to(device)
		return self

	def synchronize(self, world_size):
		cumlen = gather_tensors(self.cumlen, world_size)
		data = gather_tensors(self.data, world_size)

		# cat synchronized data
		for i in range(1, world_size):
			cumlen[i] += cumlen[i-1][-1]

		self.data = torch.cat(data, dim=0)
		self.cumlen = torch.cat(cumlen, dim=0)


