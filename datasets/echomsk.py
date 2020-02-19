import json
import html.parser

class EchomskParser(html.parser.HTMLParser):
	def __init__(self):
		super().__init__()
		self.json = None
		self.sound = None
		self.youtube = None

	def handle_starttag(self, tag, attrs):
		if tag == 'script' and ('type', 'application/ld+json') in attrs:
			self.json = True

		elif tag == 'a' and any(k == 'href' and v.endswith('.mp3') for k, v in attrs):
			self.sound = [v for k, v in attrs if k == 'href'][0]

		elif tag == 'iframe' and any(k == 'src' and 'youtube.com' in v for k, v in attrs):
			self.youtube = [v for k, v in attrs if k == 'src'][0]

	def handle_data(self, data):
		if self.json is True:
			self.json = json.loads(data)

parser = EchomskParser()
parser.feed(open('index.html').read())
print(list(parser.json.keys()))
print(parser.sound)
print(parser.youtube)
