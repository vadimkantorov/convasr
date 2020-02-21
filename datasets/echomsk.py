import os
import re
import json
import argparse
import urllib.request
import html.parser

class EchomskParser(html.parser.HTMLParser):
	def __init__(self):
		super().__init__()
		self.json = None
		self.sound = None
		self.youtube = None
		self.url = ''
		self.program = ''
		self.url_program = ''
		self.date = ''
		self.name = ''
		self.id = ''
		self.transcript = []
		self.speakers = []

	def handle_starttag(self, tag, attrs):
		if tag == 'script' and ('type', 'application/ld+json') in attrs:
			self.json = True

		elif tag == 'a' and any(k == 'href' and v.endswith('.mp3') for k, v in attrs):
			self.sound = [v for k, v in attrs if k == 'href'][0]

		elif tag == 'iframe' and any(k == 'src' and 'youtube.com' in v for k, v in attrs):
			self.youtube = [v for k, v in attrs if k == 'src'][0]

		elif tag == 'a' and any(k == 'class' and 'name_prog' in v for k, v in attrs):
			self.program = True

	def handle_data(self, data):
		normalize_speaker = lambda speaker: '.'.join(map(str.capitalize, speaker.split('.')))
		normalize_text = lambda text: ' '.join(line for line in text.strip().replace('\r\n', '\n').split('\n') if not line.isupper())

		if self.json is True:
			self.json = json.loads(data)
			self.url = self.json['mainEntityOfPage'].rstrip('/')
			self.date = self.json['datePublished'].split('T')[0]
			self.name = self.json['name'] 
			self.url_program = os.path.dirname(self.url)
			self.id = os.path.basename(self.url_program) + '_' + os.path.basename(self.url).replace('-echo', '')

			splitted = re.split(r'([А-Я]\. ?[А-Я][А-Яа-я]+)[:―] ', self.json['articleBody'])
			self.transcript = [dict(speaker = normalize_speaker(speaker), ref = normalize_text(ref)) for speaker, ref in zip(splitted[1::2], splitted[2::2])]
			self.speakers = list(sorted(set(t['speaker'] for t in self.transcript)))

		elif self.program is True and data.strip():
			self.program = data.strip()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input-path', '-i', required = True)
	parser.add_argument('--output-path', '-o')
	args = parser.parse_args()
	
	page = EchomskParser()
	page.feed(urllib.request.urlopen(args.input_path).read().decode() if 'http' in args.input_path else open(args.input_path).read())
	parsed = dict(id = page.id, name = page.name, url = page.url, date = page.date, program = page.program, url_program = page.url_program, youtube = page.youtube, sound = page.sound, transcript = page.transcript, speakers = page.speakers)
	
	if not args.output_path:
		args.output_path = args.input_path + '.json'
	if not args.output_path.endswith('.json'):
		os.makedirs(args.output_path, exist_ok = True)
		args.output_path = os.path.join(args.output_path, parsed['id'] + '.json')

	json.dump(parsed, open(args.output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
	print(args.output_path)
