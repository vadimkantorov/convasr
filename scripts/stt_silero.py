import json
import argparse

import base64
import requests
import scipy.io.wavfile 

# https://api.silero.ai/docs#operation/transcribe_transcribe_post

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i')
parser.add_argument('--output-path', '-o')
parser.add_argument('--api-token-file', default = 'sileroapitoken.txt')
args = parser.parse_args()

api_token = open(args.api_token_file).read().strip()

transcript = []
for t in json.load(open(args.input_path)):
	sample_rate, signal = scipy.io.wavfile.read(t['audio_path'])
	assert signal.dtype == 'int16' and sample_rate in [8_000, 16_000]
	
	res = requests.post('https://api.silero.ai/transcribe', json = dict(api_token = api_token, channels = 1, lang = 'ru', format = 'raw', sample_rate = sample_rate, payload = base64.b64encode(signal.tobytes()).decode())).json()
	transcript.append(dict(t, **dict(hyp = res['transcriptions'][0]['transcript'])))

json.dump(transcript, open(args.output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
print(args.output_path)
