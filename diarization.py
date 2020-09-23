import os
import argparse
import json
import torch
import audio
import models
import vis

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i')
parser.add_argument('--output-path', '-o')
parser.add_argument('--sample-rate', type = int, default = 8_000)
parser.add_argument('--window-size', type = float, default = 0.02)
parser.add_argument('--device', default = 'cuda')
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok = True)

pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia', device = args.device)#, batch_size = 8)

audio_source = ([(args.input_path, audio_name) for audio_name in os.listdir(args.input_path)] if os.path.isdir(args.input_path) else [(os.path.dirname(args.input_path), os.path.basename(args.input_path))])

for i, (input_path, audio_name) in enumerate(audio_source):
	if i > 0:
		break

	print(i, '/', len(audio_source), audio_name)

	audio_path = os.path.join(input_path, audio_name)
	transcript_path = os.path.join(args.output_path, audio_name + '.json')
	html_path = os.path.join(args.output_path, audio_name + '.html')

	signal, sample_rate = audio.read_audio(audio_path.replace('mono', 'stereo').replace('.mp3.wav', '.mp3'), sample_rate = args.sample_rate, mono = False, dtype = 'float32')
	speaker_ref, silence_ref, smoothed_ref = models.select_speaker(signal.to(args.device), silence_absolute_threshold = 0.05, silence_relative_threshold = 0.0, kernel_size_smooth_signal = 128, kernel_size_smooth_speaker = 4096)
	audio.write_audio(transcript_path + '.wav', torch.cat([signal[:, :speaker_ref.shape[-1]], smoothed_ref[:, :signal.shape[-1]].to(signal)]), sample_rate, mono = False)
	print(transcript_path + '.wav')

	transcript_ref = [dict(audio_path = audio_path, begin = float(begin), end = float(begin) + float(duration), speaker_name = str(int(speaker)), speaker = int(speaker)) for begin, duration, speaker in zip(*models.rle1d(speaker_ref.cpu()))]
	json.dump(transcript_ref, open(transcript_path, 'w'), indent = 2, sort_keys = True)
	
	#diarization = pipeline(dict(audio = audio_path))
	#transcript = [dict(audio_path = audio_path, begin = turn.start, end = turn.end, speaker_name = speaker) for turn, _, speaker in diarization.itertracks(yield_label = True)]
	#speaker_names = [None] + list(set(t['speaker_name'] for t in transcript))
	#for t in transcript:
	#	t['speaker'] = speaker_names.index(t['speaker_name'])
	#json.dump(transcript, open(transcript_path, 'w'), indent = 2, sort_keys = True)
	vis.transcript(html_path, sample_rate = args.sample_rate, mono = True, transcript = transcript_path)
