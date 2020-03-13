import os
import gzip
import json
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', '-i', required = True)
parser.add_argument('--output-path', '-o', required = True)
parser.add_argument('--min-speakers', type = int, default = 2)
parser.add_argument('--max-speakers', type = int, default = 2)
parser.add_argument('--sample', type = int, default = 10)
args = parser.parse_args()

gzopen = lambda file_path, mode = 'r': gzip.open(file_path, mode + 't') if file_path.endswith('.gz') else open(file_path, mode)
episodes = json.load(gzopen(args.input_path))
episodes = [e for e in episodes if args.min_speakers <= len(e['speakers']) <= args.max_speakers and e['sound_seconds'] > 0 and len(e['sound']) == 1]
random.seed(1)
random.shuffle(episodes)
episodes = episodes[:args.sample]

os.makedirs(args.output_path, exist_ok = True)

for e in episodes:
	transcript = [dict(audio_path = os.path.join(args.output_path, os.path.basename(e['sound'][0])), ref = t['ref'], speaker = t['speaker']) for t in e['transcript']]
	transcript_path = transcript[0]['audio_path'] + '.json'
	json.dump(transcript, open(transcript_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)

transcript_path = os.path.join(args.output_path, os.path.basename(args.output_path) + '.txt')
open(transcript_path, 'w').write('\n'.join(e['sound'][0] for e in episodes))
