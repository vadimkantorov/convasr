import re
import typing
from collections import defaultdict

ru_alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

class CharTokenizerLegacy:
    def __init__(self, alphabet: str = ru_alphabet, name: str = 'char_legacy'):
        self.name = name
        self.alphabet = alphabet
        self.unk_token = '*'
        self.punkt_token = '.'
        self.repeat_token = '2'
        self.space_token = ' '
        self.eps_token = '|'
        self.idx2char = list(alphabet) + [self.unk_token, self.punkt_token, self.repeat_token, self.space_token, self.eps_token]

        unk_idx = self.idx2char.index(self.unk_token)
        self.char2idx = defaultdict(lambda: unk_idx)
        self.char2idx.update({char: idx for idx, char in self.idx2char})

    def encode(self, sentences: typing.Iterable[str]) -> typing.List[typing.List[int]]:
        tokens = []
        for sentence in sentences:
            tokens.append([self.char2idx[char] for char in sentence])
        return tokens

    def decode(self, tokens: typing.Iterable[typing.Iterable[int]]) -> typing.List[str]:
        sentences = []
        for sentence_tokens in tokens:
            sentences.append(''.join([self.idx2char[idx] for idx in sentence_tokens]))
        return sentences
