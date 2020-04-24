import os
import json
import gzip
import random
import itertools
import argparse

from transcripts import get_duration


def dump(by_group, splits, subset_name, gz = True):
	for split_name, transcript in by_group.items():
		input_path = os.path.join(splits, f'{subset_name}_{split_name}.json') + ('.gz' if gz else '')
		with (gzip.open(input_path, 'wt') if gz else open(input_path, 'w')) as f:
			json.dump(transcript, f, indent = 2, sort_keys = True, ensure_ascii = False)
		print(input_path, '|', int(os.path.getsize(input_path) // 1e6), 'Mb',  '|', len(transcript) // 1000, 'K utt |', int(sum(get_duration(t) for t in transcript) / (60 * 60)), 'hours')

def split(by_group, groups, spec, sample_keyword = 'sample'):
	transcript = [t for group in groups for t in by_group[group]]
	random.seed(1)
	random.shuffle(transcript)
	
	cnt_ = lambda cnt, transcript: len(transcript) if cnt is None else cnt if isinstance(cnt, int) else int(len(transcript) * cnt)

	k, res = 0, {}
	for split_name, cnt in spec.items():
		if isinstance(cnt, tuple):
			cnt_0 = cnt_(cnt[0], transcript)
			shuffled = transcript[k: k + cnt_0]
			random.shuffle(shuffled)
			res[split_name] = shuffled
			res[f'{split_name}_{sample_keyword}'] = shuffled[:cnt_(cnt[1], shuffled)]
			cnt = cnt_0
		else:
			cnt = cnt_(cnt, transcript)
			res[split_name] = transcript[k : k + cnt]
		k += cnt
	return res

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--metadata', default = 'public_meta_data_v04_fx.csv.gz')
	parser.add_argument('--exclude', nargs = '*', default = ['public_exclude_file_v5.csv.gz', 'exclude_df_youtube_1120.csv.gz'])
	parser.add_argument('--benchmark', default = 'benchmark_v05_public.csv.gz')
	parser.add_argument('--output-dir', '-o', default = 'splits')
	parser.add_argument('--gzip', action = 'store_true')
	parser.add_argument('--min-kb', type = int, default = 20)
	parser.add_argument('--max-cer', default = 'clean_thresholds_cer.json')

	args = parser.parse_args()
	
	args.max_cer = json.load(open(args.max_cer))
	os.makedirs(args.output_dir, exist_ok = True)
	
	is_header_line = lambda i, l: i == 0 or ',' not in l or l[0] == ','
	gzopen = lambda file_path: gzip.open(file_path, 'rt') if file_path.endswith('.gz') else open(file_path)

	transcript = [dict(audio_path = s[-1], audio_name = os.path.basename(s[-1]), group = s[2], begin = 0.0, end = float(s[3]), ref = s[-3], file_size_kb = float(s[5])) for i, l in enumerate(gzopen(args.metadata)) if not is_header_line(i, l) for s in [l.strip().split(',')]]
	exclude = set(os.path.basename(audio_path) for f in args.exclude for i, l in enumerate(gzopen(f)) if not is_header_line(i, l) for s in [l.split(',')] for audio_path in [s[1]])
	filtered_by_cer = set(os.path.basename(audio_path) for i, l in enumerate(gzopen(args.benchmark)) if not is_header_line(i, l) for s in [l.strip().split(',')] for audio_path, group, cer in [(s[1], s[-1], float(s[-3]))] if cer <= args.max_cer[group])
	transcript = list(filter(lambda t: t.pop('file_size_kb') >= args.min_kb and t['audio_name'] not in exclude and ('_val' in t['group'] or t['audio_name'] in filtered_by_cer), transcript))

	by_group = {k : list(g) for k, g in itertools.groupby(sorted(transcript, key = lambda t: t['group']), key = lambda t: t['group'])}

	clean = split(by_group, ['voxforge_ru', 'ru_RU', 'russian_single', 'public_lecture_1', 'public_series_1'], dict(train = 0.95, val = 0.05))

	mixed_ = split(by_group, ['buriy_audiobooks_2_val', 'public_youtube700_val'], dict(val = None))
	mixed = split(by_group, ['private_buriy_audiobooks_2', 'public_youtube700', 'public_youtube1120', 'public_youtube1120_hq', 'radio_2'], dict(train = None))
	mixed['train'] += clean['train']
	random.seed(1)
	random.shuffle(mixed['train'])
	mixed['val'] = mixed_['val']
	mixed['small'] = mixed['train'][:int(0.1 * len(mixed['train']))]
	
	radio = split(by_group, ['radio_2'], dict(train = 0.9, val = 0.1))
	
	dump(radio, args.output_dir, 'radio', gz = args.gzip)
	dump(clean, args.output_dir, 'clean', gz = args.gzip)
	dump(mixed, args.output_dir, 'mixed', gz = args.gzip)
	dump(split(by_group, ['asr_calls_2_val'], dict(val = None)), args.output_dir, 'calls', gz = args.gzip)
	
	unused_groups_for_now = ['asr_public_phone_calls_2', 'asr_public_phone_calls_1', 'asr_public_stories_2', 'asr_public_stories_1', 'tts_russian_addresses_rhvoice_4voices']
