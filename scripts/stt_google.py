# pip install google-cloud-speech
# https://github.com/googleapis/python-speech
# https://googleapis.dev/python/speech/latest/index.html
# https://google-auth.readthedocs.io/en/latest/reference/google.oauth2.service_account.html

import os
import io
import json
import argparse
import scipy.io.wavfile
import grpc
import google.oauth2.service_account
import google.cloud.speech_v1
import google.cloud.speech_v1.gapic.transports.speech_grpc_transport

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i')
parser.add_argument('--output-path', '-o', default = 'data')
parser.add_argument('--verbose', action = 'store_true')
parser.add_argument('--api-key-credentials', default = 'googleapikeycredentials.json')
parser.add_argument('--lang', default = 'ru-RU')
parser.add_argument('--vendor', default = 'google')
parser.add_argument('--format', default = 'LINEAR16')
parser.add_argument('--recognition-model', default = 'phone_call', choices = ['phone_call', 'default', 'video', 'command_and_search'])
parser.add_argument('--endpoint', default = 'speech.googleapis.com:443') # google.cloud.speech_v1.SpeechClient.SERVICE_ADDRESS)
args = parser.parse_args()

os.environ.update(dict(GRPC_VERBOSITY = 'DEBUG', GRPC_TRACE = 'all') if args.verbose else {})
LocalSpeechGrpcTransport = type('LocalSpeechGrpcTransport', (google.cloud.speech_v1.gapic.transports.speech_grpc_transport.SpeechGrpcTransport, ), dict(create_channel = lambda self, address, credentials, **kwargs: grpc.secure_channel(address, credentials, **kwargs)))

credentials = google.oauth2.service_account.Credentials.from_service_account_file(args.api_key_credentials) if args.api_key_credentials else grpc.local_channel_credentials()
client_options = dict(api_endpoint = args.endpoint)
client = google.cloud.speech_v1.SpeechClient(credentials = credentials, client_options = client_options) if args.api_key_credentials else google.cloud.speech_v1.SpeechClient(transport = LocalSpeechGrpcTransport(address = args.endpoint, credentials = credentials), client_options = client_options) 
transcript = []
for t in json.load(open(args.input_path)):
	sample_rate, signal = scipy.io.wavfile.read(t['audio_path'])
	assert signal.dtype == 'int16' and sample_rate in [8_000, 16_000]
	
	pcm = io.BytesIO()
	scipy.io.wavfile.write(pcm, sample_rate, signal)
	res = client.recognize(dict(audio_channel_count = 1, encoding = args.format, sample_rate_hertz = sample_rate, language_code = args.lang, model = args.recognition_model), dict(content = pcm.getvalue()))
	hyp = res.results[0].alternatives[0].transcript
	transcript.append(dict(t, hyp = hyp))

transcript_path = os.path.join(args.output_path, os.path.basename(args.input_path) + f'.{args.vendor}.json')
json.dump(transcript, open(transcript_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
print(transcript_path)
