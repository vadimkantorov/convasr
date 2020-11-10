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
				**kwargs):
		self.drop_space_at_borders = drop_space_at_borders
		self.to_lower_case = to_lower_case
		self.collapse_char_series = collapse_char_series #collapse any amount of chars repeats to one
		self.drop_substrings = drop_substrings #drop any substring in this list from text
		self.replace_chars = replace_chars #list contain replace groups, replace group is string where first character is replacer and  other are replaceable
		self.allowed_chars = allowed_chars.replace(' ', r'\s') #all chars not included in this string will be dropped

		self.handlers = [
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
			text = re.sub('\s+', ' ', text)
			return text


class TextPreprocessor(TextProcessor):
	def __init__(self,
				 repeat_character: str = None,
				 **kwargs):
		super().__init__(**kwargs)
		self.repeat_character = repeat_character #if not none all doubled chars will be replaced by single char and repeat char

		self.handlers = [
			self.handle_strip,
			self.handle_case,
			self.handle_repeat,
			self.handle_collapse,
			self.handle_drop,
			self.handle_replace,
			self.handle_allowed
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
			self.handle_strip,
			self.handle_case,
			self.handle_collapse,
			self.handle_drop,
			self.handle_repeat,
			self.handle_replace,
			self.handle_allowed
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