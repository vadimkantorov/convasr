import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i', required = True)
parser.add_argument('--output-path', '-o', required = True)
parser.add_argument('--strip', nargs = '*', default = ['begin', 'end'])
args = parser.parse_args()

transcript = []
for info_name in os.listdir(args.input_path):
	if not info_name.endswith('.json'):
		continue
	
	info_path = os.path.join(args.input_path, info_name)
	transcript.extend(dict(audio_path = info_path.replace('.json', ''), **{k : v for k, v in t.items() if k not in args.strip} ) for t in json.load(open(info_path)).get('transcript', []))

json.dump(transcript, open(args.output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)

print(args.output_path)
