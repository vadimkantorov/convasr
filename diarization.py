import os
import argparse
import json
import torch
import vis

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i')
parser.add_argument('--output-path', '-o')
parser.add_argument('--sample-rate', type = int, default = 8_000)
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok = True)

pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')#, device = 'cpu', batch_size = 8)

audio_source = ([(args.input_path, audio_name) for audio_name in os.listdir(args.input_path)] if os.path.isdir(args.input_path) else [(os.path.dirname(args.input_path), os.path.basename(args.input_path))])

for i, (input_path, audio_name) in enumerate(audio_source):
	print(i, '/', len(audio_source), audio_name)

	audio_path = os.path.join(input_path, audio_name)
	transcript_path = os.path.join(args.output_path, audio_name + '.json')
	html_path = os.path.join(args.output_path, audio_name + '.html')

	diarization = pipeline(dict(audio = audio_path))
	
	transcript = [dict(audio_path = audio_path, begin = turn.start, end = turn.end, speaker_name = speaker) for turn, _, speaker in diarization.itertracks(yield_label = True)]
	speaker_names = [None] + list(set(t['speaker_name'] for t in transcript))
	for t in transcript:
		t['speaker'] = speaker_names.index(t['speaker_name'])

	json.dump(transcript, open(transcript_path, 'w'), indent = 2, sort_keys = True)
	vis.transcript(html_path, sample_rate = args.sample_rate, mono = True, transcript = transcript_path)
