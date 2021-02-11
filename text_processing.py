import re
import typing
import text_tokenizers

class Stemmer:
	def __init__(self, lang: str = 'ru'):
		self.lang = lang

	def __call__(self, word):
		if self.lang is None:
			return word
		else:
			## TODO replace by normal stemmer
			return word[:-3] if len(word) > 8 else word[:-2] if len(word) > 5 else word


class ProcessingPipeline:
	@staticmethod
	def make(config, name):
		pipeline_config = config['pipelines'][name]
		tokenizer_config = config['tokenizers'][pipeline_config['tokenizer']].copy()
		tokenizer = getattr(text_tokenizers, tokenizer_config.pop('class'))(**tokenizer_config)
		preprocessor_config = config['preprocess'][pipeline_config['preprocessor']]
		preprocessor = TextPreprocessor(**preprocessor_config)
		postprocessor_config = config['postprocess'][pipeline_config['postprocessor']]
		postprocessor = TextPostprocessor(**postprocessor_config)
		return ProcessingPipeline(name = name, tokenizer = tokenizer, preprocessor = preprocessor, postprocessor = postprocessor)

	def __init__(self, name: str, tokenizer, preprocessor, postprocessor):
		self.name = name
		self.tokenizer = tokenizer
		self.preprocessor = preprocessor
		self.postprocessor = postprocessor

	def preprocess(self, text):
		return self.preprocessor(text)

	def postprocess(self, text):
		return self.postprocessor(text)

	def encode(self, sentences: typing.List[str], **kwargs) -> typing.List[typing.List[int]]:
		return self.tokenizer.encode(sentences, **kwargs)

	def decode(self, sentences: typing.List[int], **kwargs) -> typing.List[typing.List[str]]:
		return self.tokenizer.decode(sentences, **kwargs)


class TextProcessor:
	def __init__(self,
				drop_space_at_borders: bool = True,
				to_lower_case: bool = True,
				collapse_char_series: bool = True,
				drop_substrings: typing.List[str] = tuple(),
				replace_chars: typing.List[str] = tuple(),
	            allowed_chars: typing.Optional[str] = None,
	            normalize_text: bool = False,
				**kwargs):
		self.drop_space_at_borders = drop_space_at_borders
		self.to_lower_case = to_lower_case
		self.collapse_char_series = collapse_char_series #collapse any amount of chars repeats to one
		self.drop_substrings = drop_substrings #drop any substring in this list from text
		self.replace_chars = replace_chars #list contain replace groups, replace group is string where first character is replacer and  other are replaceable
		self.allowed_chars = allowed_chars.replace(' ', r'\s') if allowed_chars is not None else None#all chars not included in this string will be dropped

		self.text_normalizer = TextNormalizer() if normalize_text else None

		self.handlers = [
			self.handle_normalize,
			self.handle_strip,
			self.handle_case,
			self.handle_collapse,
			self.handle_drop,
			self.handle_replace,
			self.handle_allowed
		]

	def __call__(self, text):
		for	handler in self.handlers:
			text = handler(text)
		return text

	def handle_normalize(self, text):
		if self.text_normalizer is not None:
			text = self.text_normalizer.normalize(text)
		return text

	def handle_strip(self, text):
		return text.strip() if self.drop_space_at_borders else text

	def handle_case(self, text):
		return text.lower() if self.to_lower_case else text

	def handle_collapse(self, text):
		if self.collapse_char_series:
			text = re.sub(rf'(.)\1+', rf'\g<1>', text)
		return text

	def handle_drop(self, text):
		for substring in self.drop_substrings:
			text = text.replace(substring, '')
		return text

	def handle_replace(self, text):
		for replace_group in self.replace_chars:
			assert len(replace_group), f'length of replace group should be more than 1, but get "{replace_group}"'
			replacer = replace_group[0]
			replacable = replace_group[1:]
			text = re.sub(f'[{replacable}]', replacer, text)
		return text

	def handle_allowed(self, text):
		if self.allowed_chars is None:
			return text
		else:
			text = re.sub(rf'[^{self.allowed_chars}]', '', text)
			text = re.sub('\s2', ' ', text)
			text = re.sub('\s+', ' ', text)
			return text


class TextPreprocessor(TextProcessor):
	def __init__(self,
				 repeat_character: str = None,
				 **kwargs):
		super().__init__(**kwargs)
		self.repeat_character = repeat_character #if not none all doubled chars will be replaced by single char and repeat char

		self.handlers = [
			self.handle_normalize,
			self.handle_case,
			self.handle_repeat,
			self.handle_collapse,
			self.handle_drop,
			self.handle_replace,
			self.handle_allowed,
			self.handle_strip
		]

	def handle_repeat(self, text):
		if self.repeat_character is not None:
			text = re.sub(r'(\w)\1', rf'\g<1>{self.repeat_character}', text)
		return text


class TextPostprocessor(TextProcessor):
	def __init__(self,
				 repeat_character: str = None, #if not none this character will replaced by previuos
				 **kwargs):
		super().__init__(**kwargs)
		self.repeat_character = repeat_character

		self.handlers = [
			self.handle_normalize,
			self.handle_case,
			self.handle_collapse,
			self.handle_drop,
			self.handle_repeat,
			self.handle_replace,
			self.handle_allowed,
			self.handle_strip
		]

	def handle_repeat(self, text):
		if self.repeat_character is None or len(text) == 0:
			return text
		result = [text[0]] if text[0] != self.repeat_character else []
		for i in range(1, len(text)):
			if text[i] == self.repeat_character:
				result.append(text[i-1])
			else:
				result.append(text[i])
		return ''.join(result)


class TextNormalizer:
	def __init__(self):
		self._minus = 'минус'
		self._percent = 'процент'

		self._arabic2roman = {
			1000: 'M',
			900 : 'CM',
			500 : 'D',
			400 : 'CD',
			100 : 'C',
			90  : 'XC',
			50  : 'L',
			40  : 'XL',
			10  : 'X',
			9   : 'IX',
			5   : 'V',
			4   : 'IV',
			1   : 'I'
		}

		self._roman2arabic = {}
		for i in range(1,31):
			res = ''
			x = i
			for a, r in sorted(self._arabic2roman.items(), reverse = True):
				cnt = int(x / a)
				res += r * cnt
				x -= a * cnt
			self._roman2arabic[res] = i

		self._ordinalcardinal2text = {
			0         : ('ноль', 'нулевой'),
			1         : ('один', 'первый'),
			2         : ('два', 'второй'),
			3         : ('три', 'третий'),
			4         : ('четыре', 'четвертый'),
			5         : ('пять', 'пятый'),
			6         : ('шесть', 'шестой'),
			7         : ('семь', 'седьмой'),
			8         : ('восемь', 'восьмой'),
			9         : ('девять', 'девятый'),
			10        : ('десять', 'десятый'),
			11        : ('одиннадцать', 'одиннадцатый'),
			12        : ('двенадцать', 'двенадцатый'),
			13        : ('тринадцать', 'тринадцатый'),
			14        : ('четырнадцать', 'четырнадцатый'),
			15        : ('пятнадцать', 'пятнадцатый'),
			16        : ('шестнадцать', 'шестнадцатый'),
			17        : ('семнадцать', 'семнадцатый'),
			18        : ('восемнадцать', 'восемнадцатый'),
			19        : ('девятнадцать', 'девятнадцатый'),
			20        : ('двадцать', 'двадцатый'),
			30        : ('тридцать', 'тридцатый'),
			40        : ('сорок', 'сороковой'),
			50        : ('пятьдесят', 'пятьдесятый'),
			60        : ('шестьдесят', 'шестьдесятый'),
			70        : ('семьдесят', 'семидесятый'),
			80        : ('восемьдесят', 'восемьдесятый'),
			90        : ('девяносто', 'девяностый'),
			100       : ('сто', 'сотый'),
			200       : ('двести', 'двухсотый'),
			300       : ('триста', 'трехсотый'),
			400       : ('четыреста', 'четырехсотый'),
			500       : ('пятьсот', 'пятисотый'),
			600       : ('шестьсот', 'шестисотый'),
			700       : ('семьсот', 'семисотый'),
			800       : ('восемьсот', 'восьмисотый'),
			900       : ('девятьсот', 'десятисотый'),
			1000      : ('тысяча', 'тысячный'),
			1000000   : ('миллион', 'миллионный'),
			1000000000: ('миллиард', 'миллиардный'),
		}

	def normalize(self, text):
		initial_text_start_with_space = text.startswith(' ')# bug in transcribe py when some references in y stick together
		# superscripts
		scripts = '⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉⓪①②③④⑤⑥⑦⑧⑨'
		text = re.sub(f'[{scripts}]', ' ', text)

		# percent isnt preserved
		text = text.replace('%', f' {self._percent}*')

		# ignores punct
		# extract words, numbers, ordinal numbers
		words = re.findall(r'-?\d+-\w+|-?\d+\.?\d*|[\w*]+', text)
		text = ' '.join(map(self.preprocess_word, words))
		text = ' ' + text if initial_text_start_with_space else text
		return text

	def preprocess_word(self, word):
		if word in self._roman2arabic:
			word = str(self._roman2arabic[word])

		w0, (w1, w2) = word[0], (word[1:].split('-', 1) + [''])[:2]

		is_num = (w0 == '-' or w0.isdigit()) and (not w1 or w1.isdigit())
		is_ordinal = w2 and not w2.isdigit()
		if is_num:
			word = self.arabic2text(w0 + w1, ordinal = is_ordinal)

		return word

	def arabic2text(self, num, ordinal = False):
		num = int(num)
		res = []
		if num < 0:
			res.append((self._minus, self._minus))
			num *= -1

		for a, r, in sorted(self._ordinalcardinal2text.items(), reverse = True):
			if num >= a:
				div = num // a if a > 0 else 0
				if div > 1:
					res.extend(self.arabic2text(div, ordinal = None))
				res.append(r)
				num -= div * a
				if num == 0:
					break

		return res if ordinal is None else ' '.join(
			tuple(zip(*res))[0] if not ordinal else list(tuple(zip(*res))[0])[:-1] + [res[-1][1]]
		)


# if __name__ == '__main__':
# 	import argparse
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument(
# 		'--text', default = '1-й Здорово http://echomsk.ru/programs/-echo 2.5 оу 1ого 100% XIX век XX-й век -4 13.06'
# 	)
# 	args = parser.parse_args()
# 	text_config = '/work/asr_language_model/decoder/acoustic/configs/ru_text_config.json'
# 	print('ORIG:', repr(args.text))
# 	print('NORM:', repr(normalize_text(args.text)))