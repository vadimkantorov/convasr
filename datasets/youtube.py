import os
import json
import argparse
import glob

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--split-by-parts', type=int, default=2, required=True)
	parser.add_argument('--skip-files-longer-than-hours', type=float)
	parser.add_argument('--skip-transcript-large-than-char', type=float)
	parser.add_argument('--input-path', '-i', required=True)
	parser.add_argument('--output-path', '-o', required=True)
	parser.add_argument('--strip', nargs='*', default=['begin', 'end'])
	args = parser.parse_args()


	def normalize(t):
		t['ref'] = t['ref'].replace('²', 'квадратных')
		t['ref'] = t['ref'].replace('⁶', '')
		t['ref'] = t['ref'].replace('¹', '')
		t['ref'] = t['ref'].replace('³', '')
		return t


	transcripts = []
	for i, info_path in enumerate(glob.glob(os.path.join(args.input_path, f'*/*/*.json'))):
		print(i)

		j = json.load(open(info_path))

		ref_char_count = sum([len(t.get('ref', '')) for t in j.get('transcript', [])])

		json_transcripts = [dict(
					audio_path=info_path.replace('.json', ''),
					duration=j.get('duration', 0),
					ref_char_count=ref_char_count,
					**{k: v for k, v in t.items() if k not in args.strip}) for t in j.get('transcript', [])]

		if args.skip_files_longer_than_hours:
			json_transcripts = [t for t in json_transcripts if t['duration'] / 3600.0 <= args.skip_files_longer_than_hours]

		if args.skip_transcript_large_than_char:
			json_transcripts = [t for t in json_transcripts if t['ref_char_count'] <= args.skip_transcript_large_than_char]

		transcripts.extend(json_transcripts)

	transcripts = [normalize(t) for t in transcripts]

	json.dump(transcripts, open(args.output_path, 'w'), ensure_ascii=False, indent=2, sort_keys=True)

	if args.split_by_parts:
		step = len(transcripts) // args.split_by_parts + 1
		for i in range(args.split_by_parts):
			json.dump(transcripts[i * step:(i + 1) * step],
					open(args.output_path.replace('.json', '') + f'{i}' + '.json', 'w'), ensure_ascii=False, indent=2,
					sort_keys=True)

	print(args.output_path)
