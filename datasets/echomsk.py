import re
import json
import argparse
import html.parser

class EchomskParser(html.parser.HTMLParser):
	def __init__(self):
		super().__init__()
		self.json = None
		self.sound = None
		self.youtube = None
		self.url = ''
		self.program = ''
		self.date = ''
		self.transcript = []
		self.speakers = []

	def handle_starttag(self, tag, attrs):
		if tag == 'script' and ('type', 'application/ld+json') in attrs:
			self.json = True

		elif tag == 'a' and any(k == 'href' and v.endswith('.mp3') for k, v in attrs):
			self.sound = [v for k, v in attrs if k == 'href'][0]

		elif tag == 'iframe' and any(k == 'src' and 'youtube.com' in v for k, v in attrs):
			self.youtube = [v for k, v in attrs if k == 'src'][0]

	def handle_data(self, data, speaker_regexp = re.compile(r'([А-Я]\. ?[А-Я][А-Яа-я]+)[:―] ')):
		if self.json is True:
			self.json = json.loads(data)
			self.url = self.json['mainEntityOfPage']
			self.date = self.json['datePublished']
			self.program = self.url.split('programs/')[1].split('/')[0]

			speaker = None
			for line in filter(lambda line: line and not line.isupper(), map(str.strip, self.json['articleBody'].split('\n'))):
				match = re.match(speaker_regexp, line)
				if match:
					speaker = match.group(1).replace(' ', '')
					line = line[len(match.group()):]
				
				if self.transcript and self.transcript[-1]['speaker'] == speaker:
					self.transcript[-1]['ref'] += (' ' + line)
				else:
					self.transcript.append(dict(speaker = speaker, ref = line))

			self.speakers = list(sorted(set(t['speaker'] for t in self.transcript)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input-path', '-i', required = True)
	parser.add_argument('--output-path', '-o', required = True)
	args = parser.parse_args()
	
	page = EchomskParser()
	page.feed(open(args.input_path).read())

	json.dump(dict(url = page.url, date = page.date, program = page.program, youtube = page.youtube, sound = page.sound, transcript = page.transcript, speakers = page.speakers), open(args.output_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)
