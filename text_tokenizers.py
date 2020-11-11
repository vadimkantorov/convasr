import re
import sentencepiece
import typing
from collections import defaultdict


class CharTokenizerLegacy:
    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.unk_token = '*'
        self.punkt_token = '.'
        self.repeat_token = '2'
        self.space_token = ' '
        self.eps_token = '|'
        self.idx2char = list(alphabet) + [self.unk_token, self.punkt_token, self.repeat_token, self.space_token, self.eps_token]

        self.char2idx = {char: idx for idx, char in enumerate(self.idx2char)}

        self.unk_idx = self.char2idx[self.unk_token]
        self.space_id = self.char2idx[self.space_token]
        self.eps_id = self.char2idx[self.eps_token]

    @property
    def vocab(self):
        return self.idx2char

    @property
    def vocab_size(self):
        return len(self.idx2char)

    @property
    def silence_tokens(self):
        return {self.eps_token, self.space_token}

    def is_start_word_token(self, idx):
        '''
        Returns true if new word started from this token
        '''
        return idx == self.space_id

    def encode(self, sentences: typing.List[str], **kwargs) -> typing.List[typing.List[int]]:
        tokens = []
        for sentence in sentences:
            tokens.append([self.char2idx.get(char, self.unk_idx) for char in sentence])
        return tokens

    def decode(self, tokens: typing.Iterable[typing.List[int]], **kwargs) -> typing.List[str]:
        sentences = []
        for sentence_tokens in tokens:
            sentences.append(''.join([self.idx2char[idx] for idx in sentence_tokens]))
        return sentences


class BPETokenizer:
    def __init__(self, model_path: str, name: str):
        self.bpe = sentencepiece.SentencePieceProcessor(model_file = model_path)
        self.vocab = [self.bpe.id_to_piece(idx) for idx in range(self.bpe.get_piece_size())]
        self.word_start_tokens = set(idx for idx in range(self.bpe.get_piece_size()) if '\u2581' in self.vocab[idx])

    @property
    def vocab_size(self):
        return self.bpe.get_piece_size()

    @property
    def silence_tokens(self):
        return {self.pad_id}

    def is_start_word_token(self, idx):
        '''
        Returns true if new word started from this token
        '''
        return idx in self.word_start_tokens

    @property
    def bos_id(self):
        return self.bpe.bos_id

    @property
    def eos_id(self):
        return self.bpe.eos_id

    @property
    def unk_id(self):
        return self.bpe.unk_id

    @property
    def pad_id(self):
        return self.bpe.pad_id

    def encode(self, sentences: typing.List[str], bos = False, eos = False, **kwargs) -> typing.List[typing.List[int]]:
        return self.bpe.encode(sentences, add_bos = bos, add_eos = eos)

    def decode(self, tokens: typing.List[typing.List[int]], **kwargs) -> typing.List[str]:
        return self.bpe.decode(tokens)