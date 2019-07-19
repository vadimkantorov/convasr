LABELS = "|АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ2* "

def preprocess_text(text):        
    text = text.replace('*', ' ').replace('+', ' ').replace('%', 'процент*')
    text = text.replace('ё', 'е').replace('Ё', 'Е')
    return text

def preprocess_word(w):
    if w in LATINS_2_NUM:
        return str(LATINS_2_NUM[w])
    if w.isdigit():
        return num2words(w, ordinal=False)
    elif '-' in w:
        # 123-я
        w1, w2 = w.split('-', 1)
        if w1.isdigit() and not w2.isdigit():
            return num2words(w1, ordinal=True) + w2
    return w

LATINS = """II III IV V VI VII VIII IX X
XI XII XIII XIV XV XVI XVII XVIII XIX XX
XXI XXII XXIII XXIV XXV XXVI XXVII XXVIII XXIX XXX
XXXI XXXII XXXIII XXXIV XXXV XXXVI XXXVII XXXVIII XXXIX XXXX
""".split()
LATINS_2_NUM = {x: i for i, x in enumerate(LATINS, 2)}

CARDINALS = {
    0: 'НОЛЬ',
    1: 'ОДИН*',
    2: 'ДВА*',
    3: 'ТРИ',
    4: 'ЧЕТЫРЕ',
    5: 'ПЯТЬ',
    6: 'ШЕСТЬ',
    7: 'СЕМЬ',
    8: 'ВОСЕМЬ',
    9: 'ДЕВЯТЬ',
    10: 'ДЕСЯТЬ',
    11: 'ОДИННАДЦАТЬ',
    12: 'ДВЕНАДЦАТЬ',
    13: 'ТРИНАДЦАТЬ',
    14: 'ЧЕТЫРНАДЦАТЬ',
    15: 'ПЯТНАДЦАТЬ',
    16: 'ШЕСТНАДЦАТЬ',
    17: 'СЕМНАДЦАТЬ',
    18: 'ВОСЕМНАДЦАТЬ',
    19: 'ДЕВЯТНАДЦАТЬ',
    20: 'ДВАДЦАТЬ',
    30: 'ТРИДЦАТЬ',
    40: 'СОРОК',
    50: 'ПЯТЬДЕСЯТ',
    60: 'ШЕСТЬДЕСЯТ',
    70: 'СЕМЬДЕСЯТ',
    80: 'ВОСЕМЬДЕСЯТ',
    90: 'ДЕВЯНОСТО',
    100: 'СТО',
    200: 'ДВЕСТИ',
    300: 'ТРИСТА',
    400: 'ЧЕТЫРЕСТА',
    500: 'ПЯТЬСОТ',
    600: 'ШЕСТЬСОТ',
    700: 'СЕМЬСОТ',
    800: 'ВОСЕМЬСОТ',
    900: 'ДЕВЯТЬСОТ',
    1000: 'ТЫСЯЧА*',
    1000000: 'МИЛЛИОН',
    1000000000: 'МИЛЛИАРД',
}

ORDINALS = {
    0: 'НУЛЕВОЙ',
    1: 'ПЕРВЫЙ',
    2: 'ВТОРОЙ',
    3: 'ТРЕТИЙ',
    4: 'ЧЕТВЁРТЫЙ',
    5: 'ПЯТЫЙ',
    6: 'ШЕСТОЙ',
    7: 'СЕДЬМОЙ',
    8: 'ВОСЬМОЙ',
    9: 'ДЕВЯТЫЙ',
    10: 'ДЕСЯТЫЙ',
    11: 'ОДИННАДЦАТЫЙ',
    12: 'ДВЕНАДЦАТЫЙ',
    13: 'ТРИНАДЦАТЫЙ',
    14: 'ЧЕТЫРНАДЦАТЫЙ',
    15: 'ПЯТНАДЦАТЫЙ',
    16: 'ШЕСТНАДЦАТЫЙ',
    17: 'СЕМНАДЦАТЫЙ',
    18: 'ВОСЕМНАДЦАТЫЙ',
    19: 'ДЕВЯТНАДЦАТЫЙ',
    20: 'ДВАДЦАТЫЙ',
    30: 'ТРИДЦАТЫЙ',
    40: 'СОРОКОВОЙ',
    50: 'ПЯТЬДЕСЯТЫЙ',
    60: 'ШЕСТЬДЕСЯТЫЙ',
    70: 'СЕМЬДЕСЯТЫЙ',
    80: 'ВОСЕМЬДЕСЯТЫЙ',
    90: 'ДЕВЯНОСТЫЙ',
    100: 'СОТЫЙ',
    200: 'ДВУХСОТЫЙ',
    300: 'ТРЕХСОТЫЙ',
    400: 'ЧЕТЫРЕХСОТЫЙ',
    500: 'ПЯТИСОТЫЙ',
    600: 'ШЕСТИСОТЫЙ',
    700: 'СЕМИСОТЫЙ',
    800: 'ВОСЬМИСОТЫЙ',
    900: 'ДЕВЯТИСОТЫЙ',
    1000: 'ТЫСЯЧНЫЙ',
    2000: 'ДВУХТЫСЯЧНЫЙ',
    3000: 'ТРЕХТЫСЯЧНЫЙ',
    4000: 'ЧЕТЫРЕХТЫСЯЧНЫЙ',
    5000: 'ПЯТИТЫСЯЧНЫЙ',
    6000: 'ШЕСТИТЫСЯЧНЫЙ',
    7000: 'СЕМИТЫСЯЧНЫЙ',
    8000: 'ВОСЬМИТЫСЯЧНЫЙ',
    9000: 'ДЕВЯТИТЫСЯЧНЫЙ',
    1000000: 'МИЛЛИОННЫЙ',
    1000000000: 'МИЛЛИАРДНЫЙ',
}


def num1000(num):
    parts = []
    if num >= 100:
        parts.append(num - num % 100)
        num = num % 100
    if num == 0:
        return parts
    if num % 100 < 20:
        parts.append(num)
    else:
        parts.append(num - num % 10)
        num = num % 10
        if num:
            parts.append(num)
    return parts


def num_parts(num):
    if num < 0:
        num = -num
    num = int(num)
    if num == 0:
        return [0]
    parts = []
    if len(str(num)) > 7 and str(num)[-3:] != '000':  # weird, try letter-by-letter spelling
        return [int(digit) for digit in str(num)]
    if num >= 1000000000:
        parts += num1000(num // 1000000000)
        parts.append(1000000000)
        num = num % 1000000000
    if num >= 1000000:
        parts += num1000(num // 1000000)
        parts.append(1000000)
        num = num % 1000000
    if num >= 1000:
        parts += num1000(num // 1000)
        parts.append(1000)
        num = num % 1000
    parts += num1000(num)
    return parts


def num2words(num, ordinal=False):
    num = int(num)
    words = []
    if num < 0:
        words.append('МИНУС')
        num = -num

    parts = num_parts(num)
    words += [CARDINALS[p] for p in parts]
    if ordinal:
        words[-1] = ORDINALS[parts[-1]]
    result = ' '.join(words)
    return result
