import re
import os
import json
import gzip
import argparse
import torch
import sentencepiece
import audio
import transcripts

def subset(input_path, output_path, audio_name, align_boundary_words, cer, wer, duration, gap, num_speakers, strip):
	os.makedirs(output_path, exist_ok = True)

	for transcript_name in os.listdir(input_path):
		if not transcript_name.endswith('.json'):
			continue
		transcript = json.load(open(os.path.join(input_path, transcript_name)))
		
		arg = [(k, v) for k, v in dict(cer = cer, wer = wer).items() if v]

		#(input_path + '.subset_{arg}_min{min}_max{max}.json'.format(arg = arg[0][0] if arg else '', min = arg[0][1][0] if arg else '', max = arg[0][1][1] if arg else ''))
		transcript = list(transcripts.filter(transcript, audio_name = audio_name, align_boundary_words = align_boundary_words, cer = cer, wer = wer, duration = duration, gap = gap, num_speakers = num_speakers))
	
		transcript_path = os.path.join(output_path, transcript_name)
		json.dump(transcripts.strip(transcript, strip), open(transcript_path, 'w'), ensure_ascii = False, sort_keys = True, indent = 2)
	print(output_path)

def cut(input_path, output_path, sample_rate, dilate):
	os.makedirs(output_path, exist_ok = True)
	
	transcript = json.load(open(input_path))
	signal = {}

	for t in transcript:
		audio_path = t['audio_path']

		#TODO: cannot resample wav files because of torch.int16 - better off using sox/ffmpeg directly
		signal[audio_path] = signal[audio_path] if audio_path in signal else audio.read_audio(audio_path, sample_rate, normalize = False, dtype = torch.int16)[0]
		segment_path = os.path.join(output_path, os.path.basename(audio_path) + '.{channel}-{begin:.06f}-{end:.06f}.wav'.format(**t))
		audio.write_audio(segment_path, signal[audio_path][t['channel'], int(max(t['begin'] - dilate, 0) * sample_rate) : int((t['end'] + dilate) * sample_rate)], sample_rate)
		t = dict(audio_path = segment_path, begin = 0.0, end = t['end'] - t['begin'] + 2 * dilate, ref = t['ref'], meta = t)
		json.dump(t, open(segment_path + '.json', 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
	print(output_path)

def cat(input_path, output_path):
	output_path = output_path or input_path + '.json' 
	array = lambda o: [o] if isinstance(o, dict) else o
	segments = [array(json.load(open(os.path.join(input_path, transcript_name)))) for transcript_name in os.listdir(input_path) if transcript_name.endswith('.json')]
	transcript = sum(segments, [])
	json.dump(transcript, open(output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
	print(output_path)

def du(input_path):
	transcript = json.load(open(input_path))
	print(input_path, int(os.path.getsize(input_path) // 1e6), 'Mb',  '|', len(transcript) // 1000, 'K utt |', int(sum(t['end'] - t['begin'] for t in transcript) / (60 * 60)), 'hours')

def csv2json(input_path, gz, group):
	gzopen = lambda file_path, mode = 'r': gzip.open(file_path, mode + 't') if file_path.endswith('.gz') else open(file_path, mode)
	transcript = [dict(audio_path = s[0], ref = s[1], begin = 0.0, end = float(s[2]), **(dict(group = s[0].split('/')[group]) if group >= 0 else {})) for l in gzopen(input_path) if '"' not in l for s in [l.strip().split(',')]]
	output_path = input_path + '.json' + ('.gz' if gz else '')
	json.dump(transcript, gzopen(output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
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
	float_tuple = lambda s: tuple(map(lambda ip: float(ip[1] if ip[1] else ['-inf', 'inf'][ip[0]]) , enumerate(s.split('-'))))
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--audio-name')
	cmd.add_argument('--wer', type = float_tuple)
	cmd.add_argument('--cer', type = float_tuple)
	cmd.add_argument('--duration', type = float_tuple)
	cmd.add_argument('--num-speakers', type = float_tuple)
	cmd.add_argument('--gap', type = float_tuple)
	cmd.add_argument('--align-boundary-words', action = 'store_true')
	cmd.add_argument('--strip', nargs = '*', default = ['alignment', 'words'])
	cmd.set_defaults(func = subset)
	
	cmd = subparsers.add_parser('cut')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--dilate', type = float, default = 0.0)
	cmd.add_argument('--sample-rate', '-r', type = int, default = 8_000, choices = [8_000, 16_000, 32_000, 48_000])
	cmd.set_defaults(func = cut)
	
	cmd = subparsers.add_parser('cat')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o')
	cmd.set_defaults(func = cat)

	cmd = subparsers.add_parser('csv2json')
	cmd.add_argument('input_path')
	cmd.add_argument('--gzip', dest = 'gz', action = 'store_true')
	cmd.add_argument('--group', type = int, default = 1)
	cmd.set_defaults(func = csv2json)

	cmd = subparsers.add_parser('rmoldcheckpoints')
	cmd.add_argument('experiment_id')
	cmd.add_argument('--experiments-dir', default = 'data/experiments')
	cmd.add_argument('--keepfirstperepoch', action = 'store_true')
	cmd.add_argument('--remove', action = 'store_true')
	cmd.set_defaults(func = rmoldcheckpoints)

	cmd = subparsers.add_parser('du')
	cmd.add_argument('input_path')
	cmd.set_defaults(func = du)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
