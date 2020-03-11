import os
import io
import sys
import json
import contextlib
import argparse
import scipy.io.wavfile

# git clone --recursive https://github.com/TinkoffCreditSystems/voicekit-examples.git && pip install -r voicekit-examples/python/requirements.txt
sys.path.insert(0, 'voicekit-examples/python'); import recognize

# https://voicekit.tinkoff.ru/docs/recognition
# https://voicekit.tinkoff.ru/docs/usingstt

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i')
parser.add_argument('--output-path', '-o', default = 'data')
parser.add_argument('--api-key', default = 'tinkoffapikey.txt')
parser.add_argument('--secret-key', default = 'tinkoffsecretkey.txt')
parser.add_argument('--vendor', default = 'tinkoff')
args = parser.parse_args()

os.environ.update(dict(VOICEKIT_API_KEY = open(args.api_key).read().strip(), VOICEKIT_SECRET_KEY = open(args.secret_key).read().strip()))

transcript = []
for t in json.load(open(args.input_path)):
	sample_rate, signal = scipy.io.wavfile.read(t['audio_path'])
	assert signal.dtype == 'int16' and sample_rate in [8_000, 16_000]
	
	sys.argv = ['recognize.py', t['audio_path'], '--rate', str(sample_rate), '--encoding', 'LINEAR16', '--num_channels', '1']
	stdout = io.StringIO()
	with contextlib.redirect_stdout(stdout):
		recognize.main()
	hyp = [line.replace('Transcription ', '') for line in stdout.getvalue().splitlines() if line.startswith('Transcription ')][0]
	transcript.append(dict(t, hyp = hyp))

transcript_path = os.path.join(args.output_path, os.path.basename(args.input_path) + f'.{args.vendor}.json')
json.dump(transcript, open(transcript_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
print(transcript_path)
