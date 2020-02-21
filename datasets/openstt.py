import os
import json
import gzip
import random
import itertools
import argparse

def shuffle(lines, seed = 1):
	random.seed(seed)
	random.shuffle(lines)

def dump(by_group, splits, subset_name, gz = True):
	for split_name, transcript in by_group.items():
		input_path = os.path.join(splits, f'{subset_name}_{split_name}.json') + ('.gz' if gz else '')
		with (gzip.open(input_path, 'wt') if gz else open(input_path, 'w')) as f:
			json.dump(transcript, f, indent = 2, sort_keys = True, ensure_ascii = False)
		print(input_path, '|', int(os.path.getsize(input_path) // 1e6), 'Mb',  '|', len(transcript) // 1000, 'K utt |', int(sum(t['end'] - t['begin'] for t in transcript) / (60 * 60)), 'hours')

def split(by_group, groups, spec, sample_keyword = 'sample'):
	transcript = [t for group in groups for t in by_group[group]]
	shuffle(transcript)
	
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
	parser.add_argument('--metadata', default = 'public_meta_data_v04_fx.csv')
	parser.add_argument('--exclude', nargs = '*', default = ['public_exclude_file_v5.csv', 'exclude_df_youtube_1120.csv'])
	parser.add_argument('--benchmark', default = 'benchmark_v05_public.csv')
	parser.add_argument('--splits', default = 'splits')
	parser.add_argument('--gzip', action = 'store_true')
	parser.add_argument('--min-audio-kb', type = int, default = 20)
	parser.add_argument('--benchmark-max-cer', default = dict(
		tts_russian_addresses_rhvoice_4voices = 0.2,
		private_buriy_audiobooks_2 = 0.1,
		public_youtube700 = 0.2,
		public_youtube1120 = 0.2,
		public_youtube1120_hq = 0.2,
		public_lecture_1 = 0.2,
		public_series_1 = 0.2,
		radio_2 = 0.2,
		asr_public_phone_calls_1 = 0.2,
		asr_public_phone_calls_2 = 0.2,
		asr_public_stories_1 = 0.2,
		asr_public_stories_2 = 0.2,
		ru_tts = 0.4,
		ru_ru = 0.4,
		voxforge_ru = 0.4,
		russian_single = 0.4
	), action = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: getattr(n, a.dest).update(dict([v.split('=')])))) )
	args = parser.parse_args()
	
	os.makedirs(args.splits, exist_ok = True)
	
	is_header_line = lambda i, l: i == 0 or ',' not in l or l[0] == ',' 
	transcript = [dict(audio_path = s[-1], audio_name = os.path.basename(s[-1]), group = s[2], begin = 0.0, end = float(s[3]), ref = s[-3], file_size_kb = float(s[5])) for i, l in enumerate(open(args.metadata)) if not is_header_line(i, l) for s in [l.strip().split(',')]]
	
	exclude = set(os.path.basename(audio_path) for f in args.exclude for i, l in enumerate(open(f)) if not is_header_line(i, l )for s in [l.split(',')] for audio_path in [s[1]])
	filtered_by_cer = set(os.path.basename(audio_path) for i, l in enumerate(open(args.benchmark)) if not is_header_line(i, l) for s in [l.strip().split(',')] for audio_path, group, cer in [(s[1], s[-1], float(s[-3]))] if cer <= args.benchmark_max_cer[group])
	transcript = list(filter(lambda t: t.pop('file_size_kb') >= args.min_audio_kb and t['audio_name'] not in exclude and ('_val' in t['group'] or t['audio_name'] in filtered_by_cer), transcript))
	
	by_group = {k : list(g) for k, g in itertools.groupby(sorted(transcript, key = lambda t: t['group']), key = lambda t: t['group'])}

	clean = split(by_group, ['voxforge_ru', 'ru_RU', 'russian_single', 'public_lecture_1', 'public_series_1'], dict(train = 0.95, val = 0.05))

	mixed_ = split(by_group, ['buriy_audiobooks_2_val', 'public_youtube700_val'], dict(val = None))
	mixed = split(by_group, ['private_buriy_audiobooks_2', 'public_youtube700', 'public_youtube1120', 'public_youtube1120_hq', 'radio_2'], dict(train = None))
	mixed['train'] += clean['train']
	shuffle(mixed['train'])
	mixed['val'] = mixed_['val']
	mixed['small'] = mixed['train'][:int(0.1 * len(mixed['train']))]

	dump(clean, args.splits, 'clean', gz = args.gzip)
	dump(mixed, args.splits, 'mixed', gz = args.gzip)
	#dump(split(by_group, ['asr_calls_2_val'], dict(val = None)), args.splits, 'calls', gz = args.gzip)
	
	#mixed_noradio = split(by_group, ['public_youtube700', 'public_youtube1120', 'public_youtube1120_hq'], dict(train = None))
	#radio_ = split(by_group, ['private_buriy_audiobooks_2', 'radio_2'], dict(train0 = 0.1, train1 = 0.1, train2 = 0.1, train3 = 0.1, train4 = 0.1, train5 = 0.1, train6 = 0.1, train7 = 0.1, train8 = 0.1, train9 = 0.1))
	#radio = dict(val = radio_['train0'], train1 = mixed_noradio['train'] + radio_['train1'])
	#for k in range(2, 10):
	#	radio[f'train{k}'] = radio[f'train{k-1}'] + radio_[f'train{k}']
	#dump(radio, args.splits, 'radio', gz = args.gzip)
	
	unused_groups_for_now = ['asr_public_phone_calls_2', 'asr_public_phone_calls_1', 'asr_public_stories_2', 'asr_public_stories_1', 'tts_russian_addresses_rhvoice_4voices']
