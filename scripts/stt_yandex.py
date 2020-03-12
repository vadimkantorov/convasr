import io
import os
import sys
import json
import requests
import argparse
import scipy.io.wavfile

# https://cloud.yandex.ru/docs/speechkit/stt/request

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i')
parser.add_argument('--output-path', '-o', default = 'data')
parser.add_argument('--api-key', default = 'yandexapikey.txt')
parser.add_argument('--format', default = 'lpcm')
parser.add_argument('--lang', default = 'ru-RU')
parser.add_argument('--vendor', default = 'yandex')
parser.add_argument('--endpoint', default = 'https://stt.api.cloud.yandex.net/speech/v1/stt:recognize')
args = parser.parse_args()

args.api_key = open(args.api_key).read().strip()

transcript = []
for t in json.load(open(args.input_path)):
	sample_rate, signal = scipy.io.wavfile.read(t['audio_path'])
	assert signal.dtype == 'int16' and sample_rate in [8_000, 16_000]
	hyp = requests.post(args.endpoint, headers = dict(Authorization = 'Api-Key ' + args.api_key), params = dict(lang = args.lang, sampleRateHertz = sample_rate, format = args.format, raw_results = True), data = signal.tobytes()).json()['result']
	transcript.append(dict(t, hyp = hyp))

transcript_path = os.path.join(args.output_path, os.path.basename(args.input_path) + f'.{args.vendor}.json')
json.dump(transcript, open(transcript_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
print(transcript_path)
