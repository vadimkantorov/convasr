import re
import os
import json
import argparse
import torch
import sentencepiece
import audio

def bpetrain(input_path, output_prefix, vocab_size, model_type, max_sentencepiece_length):
	sentencepiece.SentencePieceTrainer.Train(f'--input={input_path} --model_prefix={output_prefix} --vocab_size={vocab_size} --model_type={model_type}' + (f' --max_sentencepiece_length={max_sentencepiece_length}' if max_sentencepiece_length else ''))

def subset(input_path, output_path, audio_file_name, arg, min, max):
	if input_path.endswith('.csv'):
		output_path = output_path or (input_path + (audio_file_name.split('subset')[-1] if audio_file_name else '') + '.csv')
		good_audio_file_name = set(map(str.strip, open(audio_file_name)) if audio_file_name is not None else [])
		open(output_path,'w').writelines(line for line in open(input_path) if os.path.basename(line.split(',')[0]) in good_audio_file_name)
	elif input_path.endswith('.json'):
		output_path = output_path or (input_path + f'.subset_{arg}_min{min}_max{max}.txt')
		open(output_path, 'w').write('\n'.join(r['audio_file_name'] for r in json.load(open(input_path)) if (min <= r[arg] if min is not None else True) and (r[arg] < max if max is not None else True)))
	print(output_path)

def cut(input_path, output_path, sample_rate, min_duration, max_duration):
	os.makedirs(output_path, exist_ok = True)
	transcript = json.load(open(input_path))
	signal = {}

	for t in transcript:
		audio_path = t['audio_path']
		#TODO: cannot resample wav files because of torch.int16 - better off using sox/ffmpeg directly
		signal[audio_path] = signal[audio_path] if audio_path in signal else audio.read_audio(audio_path, sample_rate, normalize = False, dtype = torch.int16)[0]
		duration = t['end'] - t['begin']
		if (min_duration is None or min_duration <= duration) and (max_duration is None or duration <= max_duration):
			segment_path = os.path.join(output_path, os.path.basename(audio_path) + '.{channel}-{begin:.06f}-{end:.06f}.wav'.format(**t))
			audio.write_audio(segment_path, signal[audio_path][t['channel'], int(t['begin'] * sample_rate) : int(t['end'] * sample_rate)], sample_rate)
			t = dict(audio_path = segment_path, begin = 0.0, end = t['end'] - t['begin'], ref = t['ref'], meta = t)
			json.dump(t, open(segment_path + '.json', 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
	
def cat(input_path, output_path):
	transcript = [json.load(open(os.path.join(input_path, transcript_name))) for transcript_name in os.listdir(input_path) if transcript_name.endswith('.json')]
	json.dump(transcript, open(output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)

def fromsrt(input_path, output_path):
	os.makedirs(output_path, exist_ok = True)

	def timestamp2sec(ts):
		hh, mm, ss, msec = map(int, ts.replace(',', ':').split(':'))
		return hh * 60 * 60 + mm * 60 + ss + msec * 1e-3
	transcript = [dict(begin = timestamp2sec(begin), end = timestamp2sec(end), ref = ref) for begin, end, ref in re.findall(r'(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)\s+(.+)', open(input_path).read())]
	transcript_path = os.path.join(output_path, os.path.basename(input_path) + '.json')
	json.dump(transcript, open(transcript_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
	print(transcript_path)

def du(input_path):
	transcript = json.load(open(input_path))
	print(input_path, int(os.path.getsize(input_path) // 1e6), 'Mb',  '|', len(transcript) // 1000, 'K utt |', int(sum(t['end'] - t['begin'] for t in transcript) / (60 * 60)), 'hours')

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
	cmd.add_argument('--audio-file-name')
	cmd.add_argument('--arg', choices = ['cer', 'mer', 'der', 'wer'])
	cmd.add_argument('--min', type = float)
	cmd.add_argument('--max', type = float)
	cmd.set_defaults(func = subset)
	
	cmd = subparsers.add_parser('cut')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o', default = 'data/cut')
	cmd.add_argument('--max-duration', type = float)
	cmd.add_argument('--min-duration', type = float, default = 0.1)
	cmd.add_argument('--sample-rate', '-r', type = int, default = 8_000, choices = [8_000, 16_000, 32_000, 48_000])
	cmd.set_defaults(func = cut)
	
	cmd = subparsers.add_parser('cat')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o', default = 'data/cat.json')
	cmd.set_defaults(func = cat)

	cmd = subparsers.add_parser('fromsrt')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o', default = 'data/fromsrt')
	cmd.set_defaults(func = fromsrt)
	
	cmd = subparsers.add_parser('du')
	cmd.add_argument('input_path')
	cmd.set_defaults(func = du)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
