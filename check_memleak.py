import argparse
import json
import math
import os

import psutil
import torch
from torch.utils.data.dataloader import DataLoader

import datasets
from datasets import AudioTextDataset
from train import set_random_seed


def convert_to_dict(mem_info):
	return {
		'vms': mem_info.vms,
		'shared': mem_info.shared,
		'data': mem_info.data,
		'rss': mem_info.rss
	}


def summarize_mem_info_array(mem_info):
	keys = set(mem_info[0].keys()) if len(mem_info) > 0 else {}
	summarized = {k: sum([e[k] for e in mem_info]) for k in keys}
	return summarized


def get_mem_usage_for_parent_and_child():
	mem_info = []
	process = psutil.Process(os.getpid())
	parent_memory_info = process.memory_info()
	mem_info.append(convert_to_dict(parent_memory_info))

	current_process = psutil.Process()
	children = current_process.children(recursive=True)
	for child in children:
		process = psutil.Process(child.pid)
		children_memory_info = process.memory_info()
		#mem_info.append(convert_to_dict(children_memory_info))

	print(len(mem_info))
	return mem_info


def main(args):
	print_meminfo()

	#fill_mem_usage_for_parent_and_child()
	#return 0

	#print('i\tpercent\tfree\tavailable GB\tused GB')

	mem_used = []
	mem_used.append(psutil.virtual_memory().used / 1024**3)

	mem = psutil.virtual_memory()
	print(f'before setup - {mem.percent:5} - {mem.free / 1024 ** 3:10.2f} - {mem.available / 1024 ** 3:10.2f} - {mem.used / 1024 ** 3:10.2f}')

	normalize_text_config = 'data/normalize_text_config.json'

	normalize_text_config = json.load(open(normalize_text_config)) if os.path.exists(normalize_text_config) else {}
	labels = [datasets.Labels(datasets.Language('ru'), name = 'char', normalize_text_config = normalize_text_config)]

	ds = AudioTextDataset(args.data_path,
							labels,
							8000,
							frontend = None,
							min_duration = 0,
							max_duration = 8.0,
							time_padding_multiple = 128)

	sampler = datasets.BucketingBatchSampler(
		ds,
		batch_size = 16,
		mixing = None,
		bucket = lambda example: int(math.ceil(((example[0]['end'] - example[0]['begin']) / 0.01 + 1) / 128))
	)

	train_loader = torch.utils.data.DataLoader(
		ds,
		drop_last = False,
		collate_fn = ds.collate_fn,
		pin_memory = True,
		batch_sampler = sampler,
		num_workers = 64,
		worker_init_fn = set_random_seed,
	)

	for j in range(3):
		print(j)
		sampler.shuffle(j)
		for i, (meta, x, xlen, y, ylen) in enumerate(train_loader, start = sampler.batch_idx):
			if i % 1000 == 0:
				print_meminfo()

				mem = psutil.virtual_memory()
				print(f'{i:8} - {mem.percent:5} - {mem.free / 1024 ** 3:10.2f} - {mem.available / 1024 ** 3:10.2f} - {mem.used / 1024 ** 3:10.2f}')
				mem_used.append(mem.used / 1024**3)

			#if i > 0 and i % 5000 == 0:
			#	break

	print('difference: ', mem_used[-1] - mem_used[0])


def print_meminfo():
	mem_info = get_mem_usage_for_parent_and_child()
	mem_info = summarize_mem_info_array(mem_info)
	print({k: v / 1024 ** 3 for k, v in mem_info.items()})


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	#parser.add_argument('--data-path', default = 'data/splits/youtube/cut_val_100h.json')
	parser.add_argument('--data-path', nargs = '*', default = ['youtube/cut/cut_train.json',
		'youtube_lowpass/cut/cut_train.json',
		'youtube_gsmfr/cut/cut_train.json',
		'youtube_amrnb12200/cut/cut_train.json',
		'echomsk6000/cut2/cut2_train.json'],)

	#parser.add_argument('--data-path', default = 'youtube/cut/cut_train.json')
	args = parser.parse_args()

	main(args)
