import re
import typing


class TextProcessor:
	def __init__(self,
				drop_space_at_borders: bool = True,
				to_lower_case: bool = True,
				collapse_char_series: typing.Iterable[str] = tuple(),
				drop_substrings: typing.Iterable[str] = tuple(),
				replace_chars: typing.Iterable[str] = tuple(),
				**kwargs):
		self.drop_space_at_borders = drop_space_at_borders
		self.to_lower_case = to_lower_case
		self.collapse_char_series = collapse_char_series #collapse any amount of this chars repeats to one
		self.drop_substrings = drop_substrings #drop any substring in this list from text
		self.replace_chars = replace_chars #list contain replace groups, replace group is string where first character is replacer and  other are replaceable

		self.handers = [
			self.handle_strip,
			self.handle_case,
			self.handle_collapse,
			self.handle_drop,
			self.handle_replace
		]

	def process(self, text):
		for	handler in self.handers:
			text = handler(text)
		return text

	def handle_strip(self, text):
		return text.strip() if self.drop_space_at_borders else text

	def handle_case(self, text):
		return text.lower() if self.to_lower_case else text

	def handle_collapse(self, text):
		for char in self.collapse_char_series:
			text = re.sub(rf'({char})\1+', rf'\g<1>', text)
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


class TextPreprocessor(TextProcessor):
	def __init__(self,
				 repeat_character: str = None,
				 **kwargs):
		super().__init__(**kwargs)
		self.repeat_character = repeat_character #if not none all doubled chars will be replaced by single char and repeat char

		self.handers = [
			self.handle_strip,
			self.handle_case,
			self.handle_repeat,
			self.handle_collapse,
			self.handle_drop,
			self.handle_replace
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

		self.handers = [
			self.handle_strip,
			self.handle_case,
			self.handle_repeat,
			self.handle_collapse,
			self.handle_drop,
			self.handle_replace
		]

	def handle_repeat(self, text):
		if self.repeat_character is None:
			return text
		result = [text[0]] if text[0] != self.repeat_character else []
		for i in range(1, len(text)):
			if text[i] == self.repeat_character:
				result.append(text[i-1])
			else:
				result.append(text[i])
		return ''.join(result)