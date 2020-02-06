import json
import argparse
import sentencepiece

def bpetrain(input_path, output_prefix, vocab_size, model_type, max_sentencepiece_length):
	sentencepiece.SentencePieceTrainer.Train(f'--input={input_path} --model_prefix={output_prefix} --vocab_size={vocab_size} --model_type={model_type}' + (f' --max_sentencepiece_length={max_sentencepiece_length}' if max_sentencepiece_length else ''))

def subset1(input_path, audio_file_name, output_path):
	output_path = output_path or (input_path + (audio_file_name.split('subset')[-1] if audio_file_name else '') + '.csv')
	good_audio_file_name = set(map(str.strip, open(audio_file_name)) if audio_file_name is not None else [])
	open(output_path,'w').writelines(line for line in open(input_path) if os.path.basename(line.split(',')[0]) in good_audio_file_name)
	print(output_path)

def subset2(refhyp, arg, min, max):
	filename = refhyp + f'.subset_{arg}_min{min}_max{max}.txt'
	open(filename, 'w').write('\n'.join(r['audio_file_name'] for r in json.load(open(refhyp)) if (min <= r[arg] if min is not None else True) and (r[arg] < max if max is not None else True)))
	print(filename)

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
	
	cmd = subparsers.add_parser('subset1')
	cmd.add_argument('--input-path', '-i', required = True)
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--audio-file-name', required = True)
	cmd.set_defaults(func = subset1)

	cmd = subparsers.add_parser('subset2')
	cmd.add_argument('refhyp')
	cmd.add_argument('--arg', required = True, choices = ['cer', 'mer', 'der', 'wer'])
	cmd.add_argument('--min', type = float)
	cmd.add_argument('--max', type = float)
	cmd.set_defaults(func = subset2)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
