import json

import pandas as pd
import numpy as np
from collections import Counter
import pymorphy2


def main():

    dataset = json.load(open('unsup_dataset/unsup_dataset_transcripts_1608.csv.json'))

    def find_freq_dict(transcriptions):
        lines = []
        for line in transcriptions:
            if type(line) == str:
                lines.extend(line.strip().split(' '))
        dictionary = Counter(lines)
        dic = pd.DataFrame(dictionary.most_common(), columns=['word', 'count'])
        counts = dic['count'].sum()
        dic['freq'] = dic['count'] * 1000 / counts
        freq_dict = dic[['word', 'freq']].set_index('word').to_dict()['freq']
        return freq_dict

    freq_dict = find_freq_dict([e['ref'] for e in dataset])

    for line in dataset:
        line['score'] = np.power(np.prod([freq_dict[e] for e in line['ref'].strip().split(' ')]), -len(line['ref'].strip().split(' '))) if type(line['ref']) == str else np.nan

    dataset = list(sorted(dataset, key=lambda x: x['score']))
    print('Sataset len: ', len(dataset))

    print('Simplicity score distribution: ')
    print(pd.DataFrame({'score': [e['score'] for e in dataset]})['score'].describe(percentiles=np.arange(0, 1, 0.1)))

    dataset = dataset[int(0.1*len(dataset)):int(0.9*len(dataset))]  # clean up simple and difficult examples

    morph = pymorphy2.MorphAnalyzer()

    def has_only_dict_words(line, morph):
        for word in line.strip().split(' '):
            if len(word) > 1 and not morph.word_is_known(word, strict_ee=False):
                return False
        return True

    dataset = [e for e in dataset if has_only_dict_words(e['ref'], morph)]

    print(len(dataset))

    json.dump(dataset, open('unsup_dataset/unsup_dataset_transcripts_simple_1608_morph.csv.json', 'w'), ensure_ascii = False)


if __name__ == '__main__':

    main()