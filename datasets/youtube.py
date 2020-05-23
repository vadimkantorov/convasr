import os
import json
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i', required = True)
parser.add_argument('--output-path', '-o', required = True)
parser.add_argument('--strip', nargs = '*', default = ['begin', 'end'])
args = parser.parse_args()


for info_path in glob.glob(os.path.join(args.input_path, f'*/*.json')):

	transcript = []
	#info_path = os.path.join(args.input_path, info_name)
	transcript.extend(dict(audio_path = info_path.replace('.json', ''),
			**{k : v for k, v in t.items() if k not in args.strip}) for t in json.load(open(info_path)).get('transcript', []))
	json.dump(transcript, open(os.path.join(args.output_path, os.path.basename(info_path)), 'w'), ensure_ascii = False, indent = 2, sort_keys = True)

print(args.output_path)
