# -*- coding: utf-8 -*-

import re

PUNKT = '.'
UNK = '*'
ALPHA = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

ALPHABET = ALPHA + UNK + PUNKT

EVAL_REPLACE_GROUPS = ['её']

PHONETIC_REPLACE_GROUPS = ['оая', 'пб', 'сзц', 'вф', 'кгх', 'тд', 'чжшщ', 'еыэий', 'лр', 'ую', 'ьъ', 'нм']
VOWELS = 'аоийеёэыуюя'

ORDINALCARIDNAL2TEXT = {
	0: ('ноль', 'нулевой'),
	1: ('один', 'первый'),
	2: ('два', 'второй'),
	3: ('три', 'третий'),
	4: ('четыре', 'четвертый'),
	5: ('пять', 'пятый'),
	6: ('шесть', 'шестой'),
	7: ('семь', 'седьмой'),
	8: ('восемь', 'восьмой'),
	9: ('девять', 'девятый'),
	10: ('десять', 'десятый'),
	11: ('одиннадцать', 'одиннадцатый'),
	12: ('двенадцать', 'двенадцатый'),
	13: ('тринадцать', 'тринадцатый'),
	14: ('четырнадцать', 'четырнадцатый'),
	15: ('пятнадцать', 'пятнадцатый'),
	16: ('шестнадцать', 'шестнадцатый'),
	17: ('семнадцать', 'семнадцатый'),
	18: ('восемнадцать', 'восемнадцатый'),
	19: ('девятнадцать', 'девятнадцатsq'),
	20: ('двадцать', 'двадцатый'),
	30: ('тридцать', 'тридцатый'),
	40: ('сорок', 'сороковой'),
	50: ('пятьдесят', 'пятьдесятый'),
	60: ('шестьдесят', 'шестьдесятый'),
	70: ('семьдесят', 'семидесятый'),
	80: ('восемьдесят', 'восемьдесятый'),
	90: ('девяносто', 'девяностый'),
	100: ('сто', 'сотый'),
	200: ('двести', 'двухсотый'),
	300: ('триста', 'трехсотый'),
	400: ('четыреста', 'четырехсотый'),
	500: ('пятьсот', 'пятисотый'),
	600: ('шестьсот', 'шестисотый'),
	700: ('семьсот', 'семисотый'),
	800: ('восемьсот', 'восьмисотый'),
	900: ('девятьсот', 'десятисотый'),
	1000: ('тысяча', 'тысячный'),
	1000000: ('миллион', 'миллионный'),
	1000000000: ('миллиард', 'миллиардный'),
}

ARABIC2ROMAN = {
	1000: 'M',
	900: 'CM',
	500: 'D',
	400: 'CD',
	100: 'C',
	90: 'XC',
	50: 'L',
	40: 'XL',
	10: 'X',
	9: 'IX',
	5: 'V',
	4: 'IV',
	1: 'I'
}

INFLECTIONS = list(
	sorted([
		'а',
		'я',
		'ы',
		'и',
		'ое',
		'ее',
		'ой',
		'ые',
		'ие',
		'ый',
		'ий',
		'ать',
		'ать',
		'ять',
		'оть',
		'еть',
		'уть'
		'у'
		'ю',
		'ем',
		'им',
		'ешь',
		'ишь',
		'ете',
		'ите',
		'ет',
		'ит',
		'ут',
		'ют',
		'ят',
		'ал',
		'ял',
		'ала',
		'яла',
		'али',
		'яли',
		'ол',
		'ел',
		'ола',
		'ела',
		'оли',
		'ели',
		'ул',
		'ула',
		'ули',
		'еми',
		'емя',
		'а',
		'ам',
		'ами',
		'ас',
		'am',
		'ax',
		'ая',
		'е',
		'её',
		'ей',
		'ем',
		'ex',
		'ею',
		'ёт',
		'ёте',
		'ёх',
		'ёшь',
		'и',
		'ие',
		'ий',
		'им',
		'ими',
		'ит',
		'ите',
		'их',
		'ишь',
		'ию',
		'м',
		'ми',
		'мя',
		'о',
		'ов',
		'ого',
		'ое',
		'оё',
		'ой',
		'ом',
		'ому',
		'ою',
		'cm',
		'у',
		'ум',
		'умя',
		'ут',
		'ух',
		'ую',
		'шь'
	],
			key = len,
			reverse = True)
)


def roman2arabic(x):
	res = ''
	for a, r in sorted(ARABIC2ROMAN.items(), reverse = True):
		cnt = int(x / a)
		res += r * cnt
		x -= a * cnt
	return res


ROMAN2ARABIC = {roman2arabic(i): i for i in range(1, 31)}

MINUS = 'минус'
PERCENT = 'процент'


def arabic2text(num, ordinal = False):
	num = int(num)
	res = []
	if num < 0:
		res.append((MINUS, MINUS))
		num *= -1

	for a, r, in sorted(ORDINALCARIDNAL2TEXT.items(), reverse = True):
		if num >= a:
			div = num // a if a > 0 else 0
			if div > 1:
				res.extend(arabic2text(div, ordinal = None))
			res.append(r)
			num -= div * a
			if num == 0:
				break

	return res if ordinal is None else ' '.join(
		tuple(zip(*res))[0] if not ordinal else list(tuple(zip(*res))[0])[:-1] + [res[-1][1]]
	)


def preprocess_word(w):
	if w in ROMAN2ARABIC:
		w = str(ROMAN2ARABIC[w])

	w0, (w1, w2) = w[0], (w[1:].split('-', 1) + [''])[:2]

	is_num = (w0 == '-' or w0.isdigit()) and (not w1 or w1.isdigit())
	is_ordinal = w2 and not w2.isdigit()
	if is_num:
		w = arabic2text(w0 + w1, ordinal = w2)

	return w


def normalize_text(text, remove_unk = True):
	# unk is removed from input
	text = text.replace('*', '')

	# superscripts
	superscripts = '⁰¹²³⁴⁵⁶⁷⁸⁹'
	text = re.sub(f'[{superscripts}]', ' ', text)

	# percent isnt preserved
	text = text.replace('%', f' {PERCENT}*')

	# ignores punct
	# extract words, numbers, ordinal numbers
	words = re.findall(r'-?\d+-\w+|-?\d+\.?\d*|[\w*]+', text)
	text = ' '.join(map(preprocess_word, words))

	text = text.lower()

	# replace unk characters by star
	text = re.sub(f'[^{ALPHA} ]', '*', text)

	return text


def stem(word, inflections = [], inflection = False):
	if not inflections:
		stem = word[:-3] if len(word) > 8 else word[:-2] if len(word) > 5 else word
	
	else:
		for infl in (inflections if len(word) > 5 else []):
			if word.endswith(infl):
				stem = word[:-len(infl)]
				break
	
	infl = word[len(stem):]
	return (stem, infl) if inflection else stem


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'text', default = '1-й Здорово http://echomsk.ru/programs/-echo 2.5 оу 1ого 100% XIX век XX-й век -4 13.06'
	)
	args = parser.parse_args()
	print('ORIG:', repr(args.text))
	print('NORM:', repr(normalize_text(args.text)))
