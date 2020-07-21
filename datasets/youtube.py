import os
import json
import argparse
import glob


def main(args):
	transcripts = []
	for i, info_path in enumerate(glob.glob(os.path.join(args.input_path, '*.json'))):
		print(i)

		j = json.load(open(info_path))

		total_ref_len = sum(len(t.get('ref', '')) for t in j.get('transcript', []))

		if j.get('duration', 0) / 3600.0 >= args.skip_files_longer_than_hours:
			continue

		if total_ref_len > args.skip_transcript_large_than_char:
			continue

		transcripts_ = [dict(
				audio_path=info_path.replace('.json', ''),
				**{k: v for k, v in t.items() if k not in args.strip})
			for t in j.get('transcript', [])]
		transcripts_ = [t for t in transcripts_ if t['end'] <= args.skip_transcript_after_seconds]
		transcripts.extend(transcripts_)

	json.dump(transcripts, open(args.output_path, 'w'), ensure_ascii=False, indent=2, sort_keys=True)
	if args.split_by_parts:
		step = len(transcripts) // args.split_by_parts + 1
		for i in range(args.split_by_parts):
			json.dump(transcripts[i * step:(i + 1) * step],
					open(args.output_path.replace('.json', '') + f'{i}' + '.json', 'w'), ensure_ascii=False, indent=2,
					sort_keys=True)
	print(args.output_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--split-by-parts', type=int, default=2, required=True)
	parser.add_argument('--skip-files-longer-than-hours', type=float, default=float('inf'))
	parser.add_argument('--skip-transcript-large-than-char', type=float, default=float('inf'))
	parser.add_argument('--skip-transcript-after-seconds', type=float, default=float('inf'))
	parser.add_argument('--input-path', '-i', required=True)
	parser.add_argument('--output-path', '-o', required=True)
	parser.add_argument('--strip', nargs='*', default=[])
	args = parser.parse_args()
	main(args)
