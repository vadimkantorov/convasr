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


def main():
	mem_used = []
	mem_used.append(psutil.virtual_memory().used / 1024**3)
	normalize_text_config = 'data/normalize_text_config.json'

	normalize_text_config = json.load(open(normalize_text_config)) if os.path.exists(normalize_text_config) else {}
	labels = [datasets.Labels(datasets.Language('ru'), name = 'char', normalize_text_config = normalize_text_config)]

	ds = AudioTextDataset([args.data_path],
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
		num_workers = 16,
		worker_init_fn = set_random_seed,
	)

	sampler.shuffle(1)
	for i, (meta, x, xlen, y, ylen) in enumerate(train_loader, start = sampler.batch_idx):
		if i % 1000 == 0:
			mem = psutil.virtual_memory()
			print(
				f'{i:8} - {mem.percent:5} - {mem.free / 1024 ** 3:10.2f} - {mem.available / 1024 ** 3:10.2f} - {mem.used / 1024 ** 3:10.2f}'
			)
			mem_used.append(mem.used / 1024**3)

		if i > 0 and i % 5000 == 0:
			break

	print('difference: ', mem_used[-1] - mem_used[0])


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-path', default = 'data/splits/youtube/cut_val_100h.json')
	args = parser.parse_args()

	main(args)
