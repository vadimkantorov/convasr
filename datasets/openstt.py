import os
import base64
import gzip
import random
import itertools
import argparse

def samples_html(by_source, samples_html_path, K = 10):
	f = open(samples_html_path, 'w')
	f.write('<html><meta charset="UTF-8"><body>')
	for source, lines in sorted(by_source.items()):
		f.write(f'<h1>{source}</h1>')
		f.write('<table>')
		lines = lines[:]
		shuffle(lines)
		for i in range(K):
				splitted = lines[i].split(',')
				filename = splitted[-1].strip()
				transcript = splitted[-3]
				try:
					encoded = base64.b64encode(open(filename, 'rb').read()).decode('utf-8').replace('\n', '')
				except:
					f.write(f'<tr><td>file not found: {filename}</td></tr>')
					break
				f.write(f'<tr><td>{filename}</td><td><audio controls src="data:audio/wav;base64,{encoded}"/></td><td>{transcript}</td></tr>\n')
		f.write('</table>')

def dump(by_source, splits_dir, subset_name, gz = True):
	os.makedirs(splits_dir, exist_ok = True)
	for split_name, subset in by_source.items():
		fname = os.path.join(splits_dir, f'{subset_name}_{split_name}.csv')
		f = (gzip.open(fname + '.gz', 'wt') if gz else open(fname, 'w'))
		print(fname, '\t\tutterances:', len(subset) // 1000, 'K  hours:', int(sum(float(line.split(',')[3]) for line in subset) / 3600))
		f.write('\n'.join(','.join([s[-1], s[-3], s[3]]) for l in subset for s in [l.split(',')]))

def shuffle(lines, seed = 1):
	random.seed(seed)
	random.shuffle(lines)

def split(by_source, sources, spec, sample_keyword = 'sample'):
	lines = [l for source in sources for l in by_source[source]]
	res = {}
	shuffle(lines)
	cnt_ = lambda cnt, lines: len(lines) if cnt is None else cnt if isinstance(cnt, int) else int(len(lines) * cnt)

	k = 0
	for split_name, cnt in spec.items():
		if isinstance(cnt, tuple):
			cnt_0 = cnt_(cnt[0], lines)
			shuffled = lines[k: k + cnt_0]
			random.shuffle(shuffled)
			res[split_name] = shuffled
			res[f'{split_name}_{sample_keyword}'] = shuffled[:cnt_(cnt[1], shuffled)]
			cnt = cnt_0
		else:
			cnt = cnt_(cnt, lines)
			res[split_name] = lines[k : k + cnt]
		k += cnt

	return res

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--metadata', default = 'public_meta_data_v04_fx.csv')
	parser.add_argument('--exclude', nargs = '*', default = ['public_exclude_file_v5.csv', 'exclude_df_youtube_1120.csv'])
	parser.add_argument('--benchmark', default = 'benchmark_v05_public.csv')
	parser.add_argument('--samples', default = 'open_stt_splits.html')
	parser.add_argument('--splits', default = 'splits')
	parser.add_argument('--gzip', action = 'store_true')
	parser.add_argument('--min_kb', type = int, default = 20)
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

	meta = { os.path.basename(s[-1].strip()) : (s[2], l.strip()) for l in open(args.metadata) for s in [l.split(',')] if s[0] and float(s[5]) >= args.min_kb } 
	exclude = set(os.path.basename(s[1]) for f in args.exclude for l in open(f) for s in [l.split(',')] if s[0])
	benchmark = set(os.path.basename(s[1]) for i, l in enumerate(open(args.benchmark)) if i > 0 and ',' in l for s in [l.split(',')] if float(s[-3]) <= args.benchmark_max_cer[s[-1].strip()])
	good = { k : meta[k] for k in meta.keys() - exclude if k in benchmark or '_val' in meta[k][0] }
	by_source = {k : [t[1] for t in g] for k, g in itertools.groupby(sorted([(source, line) for k, (source, line) in good.items()], key = lambda t: t[0]), key = lambda t: t[0])}
	samples_html(by_source, args.samples)

	clean = split(by_source, ['voxforge_ru', 'ru_RU', 'russian_single', 'public_lecture_1', 'public_series_1'], dict(train = 0.95, val = 0.05))

	mixed_ = split(by_source, ['buriy_audiobooks_2_val', 'public_youtube700_val'], dict(val = None))
	mixed = split(by_source, ['private_buriy_audiobooks_2', 'public_youtube700', 'public_youtube1120', 'public_youtube1120_hq', 'radio_2'], dict(train = None))
	mixed['train'] += clean['train']
	shuffle(mixed['train'])
	mixed['val'] = mixed_['val']
	mixed['small'] = mixed['train'][:int(0.1 * len(mixed['train']))]

	calls = split(by_source, ['asr_calls_2_val'], dict(val = None))
	
	mixed_noradio = split(by_source, ['public_youtube700', 'public_youtube1120', 'public_youtube1120_hq'], dict(train = None))
	radio_ = split(by_source, ['private_buriy_audiobooks_2', 'radio_2'], dict(train0 = 0.1, train1 = 0.1, train2 = 0.1, train3 = 0.1, train4 = 0.1, train5 = 0.1, train6 = 0.1, train7 = 0.1, train8 = 0.1, train9 = 0.1))
	radio['val'] = radio_['train0']
	radio['train1'] = mixed_noradio['train'] + radio_['train1']
	for k in range(2, 10):
		radio[f'train{k}'] = radio[f'train{k-1}'] + radio_['train{k}']
	
	
	dump(clean, args.splits, 'clean', gz = args.gzip)
	dump(mixed, args.splits, 'mixed', gz = args.gzip)
	dump(calls, args.splits, 'calls', gz = args.gzip)
	dump(radio, args.splits, 'radio', gz = args.gzip)
	
	unused_sources_for_now = ['asr_public_phone_calls_2', 'asr_public_phone_calls_1', 'asr_public_stories_2', 'asr_public_stories_1', 'tts_russian_addresses_rhvoice_4voices']
