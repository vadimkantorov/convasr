import re
import os
import json
import gzip
import argparse
import subprocess
import torch
import sentencepiece
import audio
import transcripts
import datasets
import metrics
import ru as lang
from tqdm import tqdm

def subset(input_path, output_path, audio_name, align_boundary_words, cer, wer, duration, gap, unk, num_speakers):
	cat = output_path.endswith('.json')
	meta = dict(align_boundary_words = align_boundary_words, cer = cer, wer = wer, duration = duration, gap = gap, unk = unk, num_speakers = num_speakers)
	transcript_cat = []
	for transcript_name in os.listdir(input_path):
		if not transcript_name.endswith('.json'):
			continue
		transcript = json.load(open(os.path.join(input_path, transcript_name)))
		transcript = [dict(meta = meta, **t) for t in transcripts.prune(transcript, audio_name = audio_name, **meta)]
		transcript_cat.extend(transcript)

		if not cat:
			os.makedirs(output_path, exist_ok = True)
			json.dump(transcript, open(os.path.join(output_path, transcript_name), 'w'), ensure_ascii = False, sort_keys = True, indent = 2)
	if cat:
		json.dump(transcript_cat, open(output_path, 'w'), ensure_ascii = False, sort_keys = True, indent = 2)
	print(output_path)

def cut(input_path, output_path, sample_rate, mono, dilate, strip, strip_prefix, audio_backend):
	os.makedirs(output_path, exist_ok = True)
	transcript_cat = []
	
	transcript = json.load(open(input_path))
	prev_audio_path, signal = None, None

	for t in tqdm(sorted(transcript, key = lambda t: t['audio_path'])):
		audio_path = t['audio_path']
		signal = audio.read_audio(audio_path, sample_rate, normalize = False, backend = audio_backend)[0] if audio_path != prev_audio_path else signal
		t['channel'] = 0 if len(signal) == 1 else None if mono else t.get('channel')
		segment = signal[slice(t['channel'], 1 + t['channel']) if t['channel'] is not None else ..., int(max(t['begin'] - dilate, 0) * sample_rate) : int((t['end'] + dilate) * sample_rate)]
		segment_path = os.path.join(output_path, os.path.basename(audio_path) + '.{channel}-{begin:.06f}-{end:.06f}.wav'.format(**t))
		audio.write_audio(segment_path, segment, sample_rate, mono = True)
		if strip_prefix:
			segment_path = segment_path.replace('data/', '')
		t = dict(audio_path = segment_path, channel = 0 if len(signal) == 1 else None, begin = 0.0, end = segment.shape[-1] / sample_rate, speaker = t.pop('speaker', None), ref = t.pop('ref'), hyp = t.pop('hyp', None), cer = t.pop('cer', None), alignment = t.pop('alignment', {}), words = t.pop('words', {}), meta = t)
		prev_audio_path = audio_path
		transcript_cat.append(t)

	json.dump(transcripts.strip(transcript_cat, strip),open(os.path.join(output_path, os.path.basename(output_path) + '.json'), 'w'), ensure_ascii = False, sort_keys = True, indent = 2)
	print(output_path)

def cat(input_path, output_path):
	transcript_paths = [transcript_path for transcript_path in input_path if transcript_path.endswith('.json')] + [os.path.join(transcript_dir, transcript_name) for transcript_dir in input_path if os.path.isdir(transcript_dir) for transcript_name in os.listdir(transcript_dir) if transcript_name.endswith('.json')]

	array = lambda o: [o] if isinstance(o, dict) else o
	transcript = sum([array(json.load(open(transcript_path))) for transcript_path in transcript_paths], [])

	json.dump(transcript, open(output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
	print(output_path)

def du(input_path):
	transcript = json.load(open(input_path))
	print(input_path, int(os.path.getsize(input_path) // 1e6), 'Mb',  '|', len(transcript) // 1000, 'K utt |', int(sum(t['end'] - t['begin'] for t in transcript) / (60 * 60)), 'hours')

def csv2json(input_path, gz, group, reset_duration):
	gzopen = lambda file_path, mode = 'r': gzip.open(file_path, mode + 't') if file_path.endswith('.gz') else open(file_path, mode)
	def duration(audio_name):
		begin, end = map(float, os.path.splitext(audio_name)[0].split('_')[-2:])
		return end - begin
	
	transcript = [dict(audio_path = s[0], ref = s[1], begin = 0.0, end = float(s[2]) if not reset_duration else duration(os.path.basename(s[0])), **(dict(group = s[0].split('/')[group]) if group >= 0 else {})) for l in gzopen(input_path) if '"' not in l for s in [l.strip().split(',')]]
	output_path = input_path + '.json' + ('.gz' if gz else '')
	json.dump(transcript, gzopen(output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
	print(output_path)

def diff(ours, theirs, key, output_path):
	transcript_ours = {t['audio_file_name'] : t for t in json.load(open(ours))}
	transcript_theirs = {t['audio_file_name'] : t for t in json.load(open(theirs))}
	
	d = list(sorted([dict(audio_name = audio_name, diff = ours[key] - theirs[key], ref = ours['ref'], hyp_ours = ours['hyp'], hyp_thrs = theirs['hyp'])  for audio_name in transcript_ours for ours, theirs in [(transcript_ours[audio_name], transcript_theirs[audio_name])]], key = lambda d: d['diff'], reverse = True))
	json.dump(d, open(output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
	print(output_path)

def rmoldcheckpoints(experiments_dir, experiment_id, keepfirstperepoch, remove):
	assert keepfirstperepoch
	experiment_dir = os.path.join(experiments_dir, experiment_id)

	def parse(ckpt_name):
		epoch = ckpt_name.split('epoch')[1].split('_')[0]
		iteration = ckpt_name.split('iter')[1].split('.')[0]
		return int(epoch), int(iteration), ckpt_name
	ckpts = list(sorted(parse(ckpt_name) for ckpt_name in os.listdir(experiment_dir) if 'checkpoint_' in ckpt_name and ckpt_name.endswith('.pt')))
	
	if keepfirstperepoch:
		keep = [ckpt_name for i, (epoch, iteration, ckpt_name) in enumerate(ckpts) if i == 0 or epoch != ckpts[i - 1][0] or epoch == ckpts[-1][0]]
	rm = list(sorted(set(ckpt[-1] for ckpt in ckpts) - set(keep)))
	print('\n'.join(rm))

	for ckpt_name in (rm if remove else []):
		os.remove(os.path.join(experiment_dir, ckpt_name))

def bpetrain(input_path, output_prefix, vocab_size, model_type, max_sentencepiece_length):
	sentencepiece.SentencePieceTrainer.Train(f'--input={input_path} --model_prefix={output_prefix} --vocab_size={vocab_size} --model_type={model_type}' + (f' --max_sentencepiece_length={max_sentencepiece_length}' if max_sentencepiece_length else ''))

def normalize(input_path, dry = True):
	labels = datasets.Labels(lang)
	for transcript_path in input_path:
		with open(transcript_path) as f:
			transcript = json.load(f)
		for t in transcript:
			if 'ref' in t:
				t['ref'] = labels.postprocess_transcript(lang.normalize_text(t['ref']))
			if 'hyp' in t:
				t['hyp'] = labels.postprocess_transcript(lang.normalize_text(t['hyp']))
			
			if 'ref' in t and 'hyp' in t:
				t['cer'] = t['cer'] if 'cer' in t else metrics.cer(t['hyp'], t['ref'])
				t['wer'] = t['wer'] if 'wer' in t else metrics.wer(t['hyp'], t['ref'])

		if not dry:
			json.dump(transcript, open(transcript_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
		else:
			return transcript

def summary(input_path, keys):
	transcript = normalize([input_path], dry = True)
	#transcript = json.load(open(input_path))
	print(input_path)
	for k in keys:
		val = torch.FloatTensor([t[k] for t in transcript if t.get(k) is not None])
		print('{k}: {v:.02f}'.format(k = k, v = float(val.mean())))
	print()

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
	cmd.add_argument('--audio-name')
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
	cmd.add_argument('--strip-prefix', action = 'store_true')
	cmd.add_argument('--audio-backend', default = 'ffmpeg', choices = ['sox', 'ffmpeg'])
	cmd.set_defaults(func = cut)
	
	cmd = subparsers.add_parser('cat')
	cmd.add_argument('--input-path', '-i', nargs = '+')
	cmd.add_argument('--output-path', '-o')
	cmd.set_defaults(func = cat)

	cmd = subparsers.add_parser('csv2json')
	cmd.add_argument('input_path')
	cmd.add_argument('--gzip', dest = 'gz', action = 'store_true')
	cmd.add_argument('--group', type = int, default = 0)
	cmd.add_argument('--reset-duration', action = 'store_true')
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

	cmd = subparsers.add_parser('normalize')
	cmd.add_argument('input_path', nargs = '+')
	cmd.add_argument('--dry', action = 'store_true')
	cmd.set_defaults(func = normalize)

	cmd = subparsers.add_parser('summary')
	cmd.add_argument('input_path')
	cmd.add_argument('--keys', nargs = '+', default = ['cer', 'wer'])
	cmd.set_defaults(func = summary)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
