import os
import json
import argparse
import base64
import requests
import scipy.io.wavfile 

# https://api.silero.ai/docs#operation/transcribe_transcribe_post

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i')
parser.add_argument('--output-path', '-o', default = 'data')
parser.add_argument('--lang', default = 'ru')
parser.add_argument('--api-token', default = 'sileroapitoken.txt')
parser.add_argument('--vendor', default = 'silero')
parser.add_argument('--endpoint', default = 'https://api.silero.ai/transcribe')
args = parser.parse_args()

args.api_token = open(args.api_token).read().strip()

transcript = []
for t in json.load(open(args.input_path)):
	sample_rate, signal = scipy.io.wavfile.read(t['audio_path'])
	assert signal.dtype == 'int16' and sample_rate in [8_000, 16_000]
	
	hyp = requests.post(args.endpoint, json = dict(api_token = args.api_token, channels = 1, lang = args.lang, format = 'raw', sample_rate = sample_rate, payload = base64.b64encode(signal.tobytes()).decode())).json()['transcriptions'][0]['transcript']
	transcript.append(dict(t, **dict(hyp = hyp)))

transcript_path = os.path.join(args.output_path, os.path.basename(args.input_path) + f'.{args.vendor}.json')
json.dump(transcript, open(transcript_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
print(transcript_path)
