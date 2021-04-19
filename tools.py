import math
import os
import re
import json
import gzip
import argparse
import itertools
import subprocess
import collections
import functools
import torch
import sentencepiece
import tqdm
import audio
import models
import transcripts
import datasets
import metrics
import random
import hashlib
import multiprocessing
import utils
import text_processing


def subset(input_path, output_path, allowed_audio_names, align_boundary_words, cer, wer, duration, gap, unk, num_speakers):
	cat = output_path.endswith('.json')
	meta = dict(
		align_boundary_words = align_boundary_words,
		cer = cer,
		wer = wer,
		duration = duration,
		gap = gap,
		unk = unk,
		num_speakers = num_speakers
	)
	transcript_cat = []
	for transcript_name in os.listdir(input_path):
		if not transcript_name.endswith('.json'):
			continue
		transcript = json.load(open(os.path.join(input_path, transcript_name)))
		transcript = [dict(meta = meta, **t) for t in transcripts.prune(transcript, allowed_audio_names = allowed_audio_names, **meta)]
		transcript_cat.extend(transcript)

		if not cat:
			os.makedirs(output_path, exist_ok = True)
			json.dump(
				transcript,
				open(os.path.join(output_path, transcript_name), 'w'),
				ensure_ascii = False,
				sort_keys = True,
				indent = 2
			)
	if cat:
		json.dump(transcript_cat, open(output_path, 'w'), ensure_ascii = False, sort_keys = True, indent = 2)
	print(output_path)


def cut_audio(output_path, sample_rate, mono, dilate, strip_prefix, audio_backend, add_sub_paths, audio_transcripts):
	audio_path_res = []
	prev_audio_path = ''
	for t in audio_transcripts:
		audio_path = t['audio_path']
		signal = audio.read_audio(audio_path, sample_rate, 
									backend = audio_backend)[0] if audio_path != prev_audio_path else signal

		if signal.numel() == 0:  # bug with empty audio files witch produce empty cut file
			print('Empty audio_path ', audio_path)
			return []

		t['channel'] = 0 if len(signal) == 1 else None if mono else t.get('channel')
		segment = signal[slice(t['channel'], 1 + t['channel']) if t['channel'] is not None else ...,
							int(max(t['begin'] - dilate, 0) * sample_rate):int((t['end'] + dilate) * sample_rate)]

		segment_file_name = os.path.basename(audio_path) + '.{channel}-{begin:.06f}-{end:.06f}.wav'.format(**t)
		digest = hashlib.md5(segment_file_name.encode('utf-8')).hexdigest()
		sub_path = [digest[-1:], digest[:2], segment_file_name] if add_sub_paths else [segment_file_name]

		segment_path = os.path.join(output_path, *sub_path)
		os.makedirs(os.path.dirname(segment_path), exist_ok = True)
		audio.write_audio(segment_path, segment, sample_rate, mono = True)

		if strip_prefix:
			segment_path = segment_path[len(strip_prefix):] if segment_path.startswith(strip_prefix) else segment_path
			t['audio_path'] = t['audio_path'][len(strip_prefix):] if t['audio_path'].startswith(strip_prefix) else \
                              t['audio_path']

		t = dict(
			audio_path = segment_path,
			audio_name = os.path.basename(segment_path),
			channel = 0 if len(signal) == 1 else None,
			begin = 0.0,
			end = segment.shape[-1] / sample_rate,
			speaker = t.pop('speaker', None),
			ref = t.pop('ref', None),
			hyp = t.pop('hyp', None),
			cer = t.pop('cer', None),
			wer = t.pop('wer', None),
			alignment = t.pop('alignment', []),
			words = t.pop('words', []),
			meta = t
		)

		prev_audio_path = audio_path
		audio_path_res.append(t)
	return audio_path_res


def cut(
	input_path, output_path, sample_rate, mono, dilate, strip, strip_prefix, audio_backend, add_sub_paths, num_workers
):
	os.makedirs(output_path, exist_ok = True)

	transcript = json.load(open(input_path))
	print('Segment count: ', len(transcript))

	transcript_by_path = {t['audio_path']: [] for t in transcript}
	for t in transcript:
		transcript_by_path[t['audio_path']].append(t)

	print('Unique audio_path count: ', len(transcript_by_path.keys()))
	with multiprocessing.pool.Pool(processes = num_workers) as pool:
		map_func = functools.partial(
			cut_audio, output_path, sample_rate, mono, dilate, strip_prefix, audio_backend, add_sub_paths
		)
		transcript_cat = []
		for ts in tqdm.tqdm(pool.imap_unordered(map_func, transcript_by_path.values())):
			transcript_cat.extend(ts)

	json.dump(
		transcripts.strip(transcript_cat, strip),
		open(os.path.join(output_path, os.path.basename(output_path) + '2.json'), 'w'),
		ensure_ascii = False,
		sort_keys = True,
		indent = 2
	)
	print(output_path)


def cat(input_path, output_path):
	transcript_paths = [transcript_path for transcript_path in input_path if transcript_path.endswith('.json')] + [
		os.path.join(transcript_dir, transcript_name) for transcript_dir in input_path if os.path.isdir(transcript_dir)
		for transcript_name in os.listdir(transcript_dir) if transcript_name.endswith('.json')
	]

	array = lambda o: [o] if isinstance(o, dict) else o
	transcript = sum([array(json.load(open(transcript_path))) for transcript_path in transcript_paths], [])

	json.dump(transcript, open(output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
	print(output_path)


def du(input_path):
	transcript = json.load(open(input_path))
	print(
		input_path,
		int(os.path.getsize(input_path) // 1e6),
		'Mb',
		'|',
		len(transcript) // 1000,
		'K utt |',
		int(sum(transcripts.compute_duration(t) for t in transcript) / (60 * 60)),
		'hours'
	)


def csv2json(input_path, gz, group, reset_begin_end, csv_sep, audio_name_pattern=None, new_sub_path=None,
		debug_short_long_records_set_begin_end_from_name=False,
		debug_short_long_records_reset_audio_path=False,
		debug_short_long_records_clean_out_ref=False,
		debug_short_long_records_output_path=None):
	""" Convert cvs transcripts file to .csv.json transcripts file. Each line in `input_path` file must have format:
		'audio_path,transcription,begin,end\n'
		csv_sep could be 'comma', representing ',', or 'tab', representing '\t'.
		audio_name_pattern - is a regex pattern, that is used, when reset_begin_end is True. It must contain at least
			two named capturing groups: (?P<begin>...) and (?P<end>...). By default, Kontur calls patter will be used.
	"""
	audio_name_regex = re.compile(audio_name_pattern) if audio_name_pattern else re.compile(
		r'(?P<begin>\d+\.?\d*)-(?P<end>\d+\.?\d*)_\d+\.?\d*_[01]_1\d{9}\.?\d*\.wav'
	)
	# default is Kontur calls pattern, match example: '198.38-200.38_2.0_0_1582594487.376404.wav'

	def begin_end(audio_name):
		match = audio_name_regex.fullmatch(audio_name)
		assert match is not None, f'audio_name {audio_name!r} must match {audio_name_regex.pattern}'
		begin, end = float(match['begin']), float(match['end'])
		assert begin < end < 10_000, 'sanity check: begin and end must be below 10_000 seconds'
		return begin, end

	def duration(audio_name):
		begin, end = begin_end(audio_name)
		return end - begin

	def channel_then_recordid(audio_path):
		return os.path.basename(audio_path).split('_')[-2] + '_' + os.path.basename(audio_path).split('_')[-1]

	csv_sep = dict(tab = '\t', comma = ',')[csv_sep]
	res = []
	for line in utils.open_maybe_gz(input_path):
		assert '"' not in line, f'{input_path!r} lines must not contain any quotation marks!'
		audio_path, ref, begin, end = line[:-1].split(csv_sep)[:4]
		transcription = dict(audio_path = audio_path, ref = ref, begin = float(begin), end = float(end))
		if reset_begin_end:
			transcription['begin'] = 0.0
			transcription['end'] = duration(os.path.basename(audio_path))
		if debug_short_long_records_set_begin_end_from_name:
			(begin, end) = begin_end(os.path.basename(audio_path))
			transcription['begin'] = begin
			transcription['end'] = end
		if debug_short_long_records_reset_audio_path:
			transcription['old_audio_path'] = audio_path
			transcription['audio_path'] = os.path.join(new_sub_path if new_sub_path else os.path.join(*os.path.split(audio_path)[:-1]),
					channel_then_recordid(audio_path))
			transcription['audio_path'] = transcription['audio_path'].replace('short_records', 'long_records')
		if debug_short_long_records_clean_out_ref:
			transcription['ref'] = ''

		# add input_path folder name to the 'group' key of each transcription
		# todo: rename --group parameter to something more sensible!
		if group >= 0:
			transcription['group'] = audio_path.split('/')[group]
		res.append(transcription)

	res.sort(key=lambda x: x['begin'])

	output_path = (debug_short_long_records_output_path if debug_short_long_records_output_path else input_path) + '.json' + ('.gz' if gz else '')
	json.dump(res, utils.open_maybe_gz(output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = False)
	print(output_path)


def diff(ours, theirs, key, output_path):
	transcript_ours = {t['audio_file_name']: t for t in json.load(open(ours))}
	transcript_theirs = {t['audio_file_name']: t for t in json.load(open(theirs))}

	d = list(
		sorted([
			dict(
				audio_name = audio_name,
				diff = ours[key] - theirs[key],
				ref = ours['ref'],
				hyp_ours = ours['hyp'],
				hyp_thrs = theirs['hyp']
			) for audio_name in transcript_ours for ours,
			theirs in [(transcript_ours[audio_name], transcript_theirs[audio_name])]
		],
				key = lambda d: d['diff'],
				reverse = True)
	)
	json.dump(d, open(output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
	print(output_path)


def rmoldcheckpoints(experiments_dir, experiment_id, keepfirstperepoch, remove):
	assert keepfirstperepoch
	experiment_dir = os.path.join(experiments_dir, experiment_id)

	def parse(ckpt_name):
		epoch = ckpt_name.split('epoch')[1].split('_')[0]
		iteration = ckpt_name.split('iter')[1].split('.')[0]
		return int(epoch), int(iteration), ckpt_name

	ckpts = list(
		sorted(
			parse(ckpt_name)
			for ckpt_name in os.listdir(experiment_dir)
			if 'checkpoint_' in ckpt_name and ckpt_name.endswith('.pt')
		)
	)

	if keepfirstperepoch:
		keep = [
			ckpt_name for i, (epoch, iteration, ckpt_name) in enumerate(ckpts)
			if i == 0 or epoch != ckpts[i - 1][0] or epoch == ckpts[-1][0]
		]
	rm = list(sorted(set(ckpt[-1] for ckpt in ckpts) - set(keep)))
	print('\n'.join(rm))

	for ckpt_name in (rm if remove else []):
		os.remove(os.path.join(experiment_dir, ckpt_name))


def bpetrain(input_path, output_prefix, vocab_size, model_type, max_sentencepiece_length):
	sentencepiece.SentencePieceTrainer.Train(
		f'--input={input_path} --model_prefix={output_prefix} --vocab_size={vocab_size} --model_type={model_type}' +
		(f' --max_sentencepiece_length={max_sentencepiece_length}' if max_sentencepiece_length else '')
	)


def transcode(input_path, output_path, ext, cmd):
	transcript = json.load(open(input_path))
	os.makedirs(output_path, exist_ok = True)
	print(cmd)
	for t in transcript:
		output_audio_path = os.path.join(output_path, os.path.basename(t['audio_path'])) + (ext or '')
		with open(t['audio_path'], 'rb') as stdin, open(output_audio_path, 'wb') as stdout:
			subprocess.check_call(cmd, stdin = stdin, stdout = stdout, shell = True)
		t['audio_path'] = output_audio_path

	output_path = os.path.join(output_path, os.path.basename(output_path) + '.json')
	json.dump(transcript, open(output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
	print(output_path)


def lserrorwords(input_path, output_path, comment_path, freq_path, sortdesc, sortasc, comment_filter, lang):
	regex = r'[ ]+-[ ]*', '-'
	freq = {
		splitted[0]: int(splitted[-1])
		for line in open(freq_path) for splitted in [re.sub(regex[0], regex[1], line).split()]
	} if freq_path else {}
	comment = {
		splitted[0]: splitted[-1].strip()
		for line in open(comment_path) for splitted in [line.split(',')] if '#' not in line and len(splitted) > 1
	} if comment_path else {}
	transcript = json.load(open(input_path))
	transcript = list(filter(lambda t: [(w.get('type') or w.get('error_tag')) for w in t['words']].count('missing_ref') <= 2, transcript))

	stem = text_processing.Stemmer(lang)
	words_ok = [w['ref'].replace(metrics.placeholder, '') for t in transcript for w in t['words'] if (w.get('type') or w.get('error_tag')) == 'ok']
	words_error = [
		w['ref'].replace(metrics.placeholder, '')
		for t in transcript
		for w in t['words']
		if (w.get('type') or w.get('error_tag')) not in ['ok', 'missing_ref']
	]
	words_error = set(ref for ref in words_error if len(ref) > 1)
	usage = {
		k: [tup[1] for tup in g]
		for k,
		g in itertools.groupby(
			sorted([(w['ref'].replace(metrics.placeholder, ''), t) for t in transcript for w in t['words']],
					key = lambda t: t[0]),
			key = lambda t: t[0]
		)
	}

	words_ok_counter = collections.Counter(map(stem, words_ok))
	words_error_counter = collections.Counter(map(stem, words_error))
	group = lambda c: stem(c[0])
	#comment = {k : ';'.join(set(c[1] for c in g if c[1])) for k, g in itertools.groupby(sorted(comment.items(), key = group), key = group)}

	#words = {ref : (ref, words_error_counter[l] - words_ok_counter[l], words_error_counter[l], words_ok_counter[l], freq.get(ref, 0), (usage.get(ref, []) + usage_placeholder)[0], (usage.get(ref, []) + usage_placeholder)[1], comment.get(ref, '')) for ref in words_error for l in [stem(ref)]}
	words = {
		ref: (
			ref,
			words_error_counter[l] - words_ok_counter[l],
			words_error_counter[l],
			words_ok_counter[l],
			freq.get(ref, 0),
			usage.get(ref, [{}])[0]['audio_name'],
			usage.get(ref, [{}])[0]['ref'],
			comment.get(ref, '')
		)
		for ref in words_error for l in [stem(ref)]
	}
	key = sortdesc or sortasc
	words = list(
		sorted(
			words.values(),
			key = lambda t: (t[-5] if key == 'diff' else (-t[2] - t[3], t[5]), t[0]),
			reverse = bool(sortdesc)
		)
	)
	words = filter(lambda tup: comment_filter in tup[-1], words)
	f = open(output_path, 'w')
	if output_path.endswith('.csv'):
		f.write('#word,diff,err,ok,freq,audioname,usage,comment\n' + '\n'.join(','.join(map(str, t)) for t in words))
	elif output_path.endswith('.json'):
		json.dump([
			dict(audio_name = audio_name, before = word, after = '') for word,
			diff,
			err,
			ok,
			freq,
			audio_name,
			usage,
			comment in words
		],
					f,
					ensure_ascii = False,
					indent = 2,
					sort_keys = True)

	print(output_path)


def wordtags(output_path, comment_path, map_tag, stop_tag):
	comment = {
		splitted[0]: splitted[-1].strip()
		for line in open(comment_path)
		for splitted in [line.split(',')]
		if '#' not in line and len(splitted) > 1 and splitted[-1].strip()
	} if comment_path else {}
	
	key = lambda t: t[1]
	value = lambda t: t[0]
	tags = { map_tag.get(k, k) : list(map(value, g)) for k, g in itertools.groupby(sorted(comment.items(), key = key), key = key)}
	tags['stop'] = tags.get('stop', []) + stop_tag
	json.dump(tags, open(output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
	print(output_path)

def processcomments(input_path, output_path, comment_path):
	transcript = json.load(open(input_path))
	comment = {
		splitted[0]: splitted[-1].strip()
		for line in open(comment_path)
		for splitted in [line.split(',')]
		if '#' not in line and len(splitted) > 1 and splitted[-1].strip()
	} if comment_path else {}

	not_word = set(k for k, v in comment.items() if v == 'naw')
	terms = set(k for k, v in comment.items() if v == 'comp' or v == 'term' or v == 'abbr')

	exclude = not_word | terms

	normalize = lambda ref: ref.replace(metrics.placeholder, '')

	print('Before filtering:', len(transcript))
	transcript = [t for t in transcript if not any(normalize(w['ref']) in exclude for w in t['words'])]
	print('After filtering:', len(transcript))

	json.dump(transcript, open(output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
	print(output_path)

	#not_word_stats = collections.defaultdict(lambda: dict(b = 0, e = 0, be = 0))
	#slices = dict(b = slice(1), e = slice(-1, None), be = slice(1, -1))
	#for t in transcript:
	#	for k, s in slices.items():
	#		for w in t['words'][s]:
	#			w_ = normalize(w['ref'])
	#			if w_ in not_word:
	#				not_word_stats[w_][k] += 1
	#print('\n'.join('{w},{be},{be_}'.format(w = w, be = not_word_stats[w]['be'], be_ = not_word_stats[w]['b'] + not_word_stats[w]['e']) for w in not_word))


def filter_dataset(input_path,
		output_path,
		duration_in_hours,
		cer,
		seed):
	dataset = transcripts.load(input_path)

	random.seed(seed)
	random.shuffle(dataset)

	print('initial set hours: ', sum(transcripts.compute_duration(t, hours=True) for t in dataset), 'hours')
	if cer:
		dataset = [e for e in dataset if e['cer'] <= cer]
		print('after cer filtering hours: ', sum(transcripts.compute_duration(t, hours=True) for t in dataset), 'hours')

	if duration_in_hours is not None:
		s = []
		set_duration = 0
		while set_duration <= duration_in_hours and len(dataset) > 0:
			t = dataset.pop()
			set_duration += transcripts.compute_duration(t, hours = True)
			s.append(t)
		dataset = s

	print('after duration filtering hours: ', sum(transcripts.compute_duration(t, hours=True) for t in dataset), 'hours')
	print(output_path)
	transcripts.save(output_path, dataset)


def split(
	input_path,
	output_path,
	test_duration_in_hours,
	val_duration_in_hours,
	microval_duration_in_hours,
	old_microval_path,
	seed
):
	transcripts_train = json.load(open(input_path))

	random.seed(seed)
	random.shuffle(transcripts_train)

	for t in transcripts_train:
		t.pop('alignment')
		t.pop('words')
		t['meta'].pop('words_hyp')
		t['meta'].pop('words_ref')

	if old_microval_path:
		old_microval = json.load(open(os.path.join(output_path, old_microval_path)))
		old_microval_pahts = set([e['audio_path'] for e in old_microval])
		transcripts_train = [e for e in transcripts_train if e['audio_path'] not in old_microval_pahts]

	for set_name, duration in [('test', test_duration_in_hours), ('val', val_duration_in_hours), ('microval', microval_duration_in_hours)]:
		if duration is not None:
			print(set_name)
			s = []
			set_duration = 0
			while set_duration <= duration:
				t = transcripts_train.pop()
				set_duration += transcripts.compute_duration(t, hours = True)
				s.append(t)
			json.dump(
				s,
				open(os.path.join(output_path, os.path.basename(output_path) + f'_{set_name}.json'), 'w'),
				ensure_ascii = False,
				sort_keys = True,
				indent = 2
			)

	json.dump(
		transcripts_train,
		open(os.path.join(output_path, os.path.basename(output_path) + '_train.json'), 'w'),
		ensure_ascii = False,
		sort_keys = True,
		indent = 2
	)

# transcript = json.load(open('data/transcripts_valset_16082020.csv.json_GreedyDecoder.json'))
#
# cer = float(torch.FloatTensor([t['cer'] for t in transcript if t['num_words'] > 0]).mean())
# wer = float(torch.FloatTensor([t['wer'] for t in transcript if t['num_words'] > 0]).mean())
# print('base', cer, wer)
#
# for k in ['words_only_proper', 'words_only_number']:
#	cer = float(torch.FloatTensor([t[k]['cer'] for t in transcript if t[k]['num_words'] > 0]).mean())
#	wer = float(torch.FloatTensor([t[k]['wer'] for t in transcript if t[k]['num_words'] > 0]).mean())
#	print(k, cer, wer)


'''
This script helps to find solution for input and output shapes of signal with frontend divisibility restrictions

example:
python tools.py find_solution_for_frontend_input_output_shapes_divisibility --start 119 --end 121 --input-time-dim-multiple 16 --output-time-dim-multiple 32
'''

def find_solution_for_frontend_input_output_shapes_divisibility(
		window_size,
		window_stride,
		sample_rate,
		start,
		end,
		input_time_dim_multiple,
		output_time_dim_multiple
):

	win_length = int(window_size * sample_rate)
	hop_length = int(window_stride * sample_rate)
	nfft = 2 ** math.ceil(math.log2(win_length))
	freq_cutoff = nfft // 2 + 1
	padding = freq_cutoff - 1  # additional_padding uses in fronted, two times for mirror and constant pad

	for i in range(start * sample_rate, end * sample_rate):
		if i % input_time_dim_multiple == 0:
			l_out = models.LogFilterBankFrontend.compute_output_shape(
					l_in=i,
					kernel_size=nfft,
					stride=hop_length,
					padding=padding,
					dilation=1)

			if l_out % output_time_dim_multiple == 0:
				print(f'Solution found: {i / sample_rate} in sec, '
				      f'input shape: {i}, output shape after frontend: {l_out}.')

	print('Finished!')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()
	cmd = subparsers.add_parser('bpetrain')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-prefix', '-o', required = True)
	cmd.add_argument('--vocab-size', default = 5000, type = int)
	cmd.add_argument('--model-type', default = 'unigram', choices = ['unigram', 'bpe', 'char', 'word'])
	cmd.add_argument('--max-sentencepiece-length', type = int, default = None)
	cmd.set_defaults(func = bpetrain)

	cmd = subparsers.add_parser('subset')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--allowed-audio-names')
	cmd.add_argument('--wer', type = transcripts.number_tuple)
	cmd.add_argument('--cer', type = transcripts.number_tuple)
	cmd.add_argument('--duration', type = transcripts.number_tuple)
	cmd.add_argument('--num-speakers', type = transcripts.number_tuple)
	cmd.add_argument('--gap', type = transcripts.number_tuple)
	cmd.add_argument('--unk', type = transcripts.number_tuple)
	cmd.add_argument('--align-boundary-words', action = 'store_true')
	cmd.set_defaults(func = subset)

	cmd = subparsers.add_parser('cut')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--dilate', type = float, default = 0.0)
	cmd.add_argument('--sample-rate', '-r', type = int, default = 8_000, choices = [8_000, 16_000, 32_000, 48_000])
	cmd.add_argument('--strip', nargs = '*', default = ['alignment', 'words'])
	cmd.add_argument('--mono', action = 'store_true')
	cmd.add_argument('--strip-prefix', type = str, default = '')
	cmd.add_argument('--audio-backend', default = 'ffmpeg', choices = ['sox', 'ffmpeg'])
	cmd.add_argument('--add-sub-paths', action = 'store_true')
	cmd.add_argument('--num-workers', type = int, default = 20)
	cmd.set_defaults(func = cut)

	cmd = subparsers.add_parser('cat')
	cmd.add_argument('--input-path', '-i', nargs = '+')
	cmd.add_argument('--output-path', '-o')
	cmd.set_defaults(func = cat)

	cmd = subparsers.add_parser('csv2json')
	cmd.add_argument('input_path')
	cmd.add_argument('--gzip', dest = 'gz', action = 'store_true')
	cmd.add_argument('--group', type = int, default = 0)
	cmd.add_argument('--reset-begin-end', action = 'store_true')
	cmd.add_argument('--debug-short-long-records-set-begin-end-from-name', action = 'store_true')
	cmd.add_argument('--debug-short-long-records-reset-audio-path', action = 'store_true')
	cmd.add_argument('--debug-short-long-records-clean-out-ref', action = 'store_true')
	cmd.add_argument('--audio-name-pattern', type = str, default = None)
	cmd.add_argument('--new-sub-path', type = str, default = None)
	cmd.add_argument('--csv-sep', default = 'tab', choices = ['tab', 'comma'])
	cmd.set_defaults(func = csv2json)

	cmd = subparsers.add_parser('diff')
	cmd.add_argument('--ours', required = True)
	cmd.add_argument('--theirs', required = True)
	cmd.add_argument('--key', default = 'cer')
	cmd.add_argument('--output-path', '-o', default = 'data/diff.json')
	cmd.set_defaults(func = diff)

	cmd = subparsers.add_parser('rmoldcheckpoints')
	cmd.add_argument('experiment_id')
	cmd.add_argument('--experiments-dir', default = 'data/experiments')
	cmd.add_argument('--keepfirstperepoch', action = 'store_true')
	cmd.add_argument('--remove', action = 'store_true')
	cmd.set_defaults(func = rmoldcheckpoints)

	cmd = subparsers.add_parser('du')
	cmd.add_argument('input_path')
	cmd.set_defaults(func = du)

	cmd = subparsers.add_parser('transcode')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o', required = True)
	cmd.add_argument('--ext', choices = ['.mp3', '.wav', '.gsm', '.raw', '.m4a', '.ogg'])
	cmd.add_argument('cmd', nargs = argparse.REMAINDER)
	cmd.set_defaults(func = transcode)

	cmd = subparsers.add_parser('lserrorwords')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o', default = 'data/error_words.csv')
	cmd.add_argument('--comment-path', '-c')
	cmd.add_argument('--freq-path', '-f')
	cmd.add_argument('--sortdesc', choices = ['diff', 'freq'])
	cmd.add_argument('--sortasc', choices = ['diff', 'freq'])
	cmd.add_argument('--comment-filter', default = '')
	cmd.add_argument('--lang', default = 'ru')
	cmd.set_defaults(func = lserrorwords)

	cmd = subparsers.add_parser('split')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--old-microval-path', type = str)
	cmd.add_argument('--test-duration-in-hours', required = True, type = float)
	cmd.add_argument('--val-duration-in-hours', required = True, type = float)
	cmd.add_argument('--microval-duration-in-hours', required = True, type = float)
	cmd.add_argument('--seed', type = int, default = 42)
	cmd.set_defaults(func = split)

	cmd = subparsers.add_parser('filter_dataset')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o', required = True)
	cmd.add_argument('--duration-in-hours', required = False, type = float)
	cmd.add_argument('--cer', required = False, type = float)
	cmd.add_argument('--seed', required = False, type = int, default = 42)
	cmd.set_defaults(func = filter_dataset)

	cmd = subparsers.add_parser('processcomments')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o', default = 'data')
	cmd.add_argument('--comment-path', '-c')
	cmd.set_defaults(func = processcomments)
	
	cmd = subparsers.add_parser('wordtags')
	cmd.add_argument('--output-path', '-o', default = 'data/word_tags.json')
	cmd.add_argument('--comment-path', '-c')
	cmd.add_argument('--map-tag', action = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: getattr(n, a.dest).update(dict([v.split('=')])))), default = dict())
	cmd.add_argument('--stop-tag', action = 'append', default = [])
	cmd.set_defaults(func = wordtags)

	cmd = subparsers.add_parser('find_solution_for_frontend_input_output_shapes_divisibility')
	cmd.add_argument('--sample-rate', type=int, default=8_000, help='for frontend')
	cmd.add_argument('--window-size', type=float, default=0.02, help='for frontend, in seconds')
	cmd.add_argument('--window-stride', type=float, default=0.01, help='for frontend, in seconds')
	cmd.add_argument('--input-time-dim-multiple', type=int, default=16)
	cmd.add_argument('--output-time-dim-multiple', type=int, default=32)
	cmd.add_argument('--start', type=int, default=118, help='time is seconds for solution search')
	cmd.add_argument('--end', type=int, default=122, help='time is seconds for solution search')
	cmd.set_defaults(func=find_solution_for_frontend_input_output_shapes_divisibility)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
