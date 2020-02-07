import os
import json
import argparse
import torch
import sentencepiece
import dataset
import vad
import segmentation

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

def cut(input_path, output_path, sample_rate, window_size, aggressiveness, min_duration, max_duration):
	os.makedirs(output_path, exist_ok = True)
	signal, sample_rate = dataset.read_audio(input_path, sample_rate, normalize = False, dtype = torch.int16)
	speech = vad.detect_speech(signal, sample_rate, window_size, aggressiveness)

	# ensure can expand and half-expand
	for c, channel in enumerate(signal):
		segments = segmentation.segment(speech[c], sample_rate = sample_rate)
		for s in segments:
			if min_duration <= s['end'] - s['begin'] <= max_duration:
				output_file_name = os.path.basename(input_path) + f'.{c}-{begin:.06f}-{end:.06f}.wav'
				dataset.write_audio(os.path.join(output_path, output_file_name), channel[s['i'] : 1 + s['j']], sample_rate)
		
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
	cmd.add_argument('--output-path', '-o', required = True)
	cmd.add_argument('--aggressiveness', type = int, choices = [0, 1, 2, 3], default = 3)
	cmd.add_argument('--window-size', type = float, choices = [0.01, 0.02, 0.03], default = 0.02)
	cmd.add_argument('--max-duration', type = float, default = 5.0)
	cmd.add_argument('--min-duration', type = float, default = 2.0)
	cmd.add_argument('--sample-rate', '-r', type = int, default = 8_000, choices = [8_000, 16_000, 32_000, 48_000])
	cmd.set_defaults(func = cut)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
