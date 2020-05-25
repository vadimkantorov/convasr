# pip install google-cloud-speech
# https://github.com/googleapis/python-speech
# https://googleapis.dev/python/speech/latest/index.html
# https://google-auth.readthedocs.io/en/latest/reference/google.oauth2.service_account.html

import os
import json
import argparse
import scipy.io.wavfile
import google.cloud.speech_v1

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i')
parser.add_argument('--output-path', '-o', default = 'data')
parser.add_argument('--api-key-credentials', default = 'googleapikeyserviceaccount.json')
parser.add_argument('--lang', default = 'ru-RU')
parser.add_argument('--vendor', default = 'google')
parser.add_argument('--endpoint', default = 'speech.googleapis.com:443') # google.cloud.speech_v1.SpeechClient.SERVICE_ADDRESS)
args = parser.parse_args()

credentials = service_account.Credentials.from_service_account_file(args.api_key_credentials) if args.api_key_credentials else {}
client = google.cloud.speech_v1.SpeechClient(credentials = credentials, client_options = dict(api_endpoint = args.api_endpoint))

transcript = []
for t in json.load(open(args.input_path)):
	sample_rate, signal = scipy.io.wavfile.read(t['audio_path'])
	assert signal.dtype == 'int16' and sample_rate in [8_000, 16_000]

	req_config = dict(
		encoding = google.cloud.speech_v1.enums.RecognitionConfig.AudioEncoding.LINEAR16, 
		sample_rate_hertz = sample_rate,
		language_code = args.lang
	)
	
	req_audio = google.cloud.speech_v1.types.RecognizeRequest(audio_content = open(t['audio_path'], 'rb').read())
	
	res = client.recognize(req_config, req_audio)
	hyp = res.alternatives[0].transcript

	transcript.append(dict(t, hyp = hyp))

transcript_path = os.path.join(args.output_path, os.path.basename(args.input_path) + f'.{args.vendor}.json')
json.dump(transcript, open(transcript_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
print(transcript_path)
