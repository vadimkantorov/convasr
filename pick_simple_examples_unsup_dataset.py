import json

import pandas as pd
import numpy as np
from collections import Counter
import pymorphy2


def has_only_dict_words(line, morph):
    for word in line.strip().split(' '):
        if len(word) > 1 and not morph.word_is_known(word, strict_ee=False):
            return False
    return True


def main():
    dataset = json.load(open('unsup_dataset/unsup_dataset_transcripts.csv.json'))

    freq_dict = find_freq_dict([e['ref'] for e in dataset])

    for line in dataset:
        line['score'] = np.power(np.prod([freq_dict[e] for e in line['ref'].strip().split(' ')]), -len(line['ref'].strip().split(' '))) if type(line['ref']) == str else np.nan

    dataset = list(sorted(dataset, key=lambda x: x['score']))
    print(len(dataset))

    print(pd.DataFrame({'score': [e['score'] for e in dataset]})['score'].describe(percentiles=np.arange(0, 1, 0.1)))
    #for i in dataset[120000:122000]:
    #    print(i)

    #with open('unsup_scores2.csv', 'a+') as file:
    #    for i in dataset[1000000:]:
    #        file.write(f"{i['ref']}\t{i['score']}\n")

    dataset = dataset[100000:1100000]

    morph = pymorphy2.MorphAnalyzer()
    dataset = [e for e in dataset if has_only_dict_words(e['ref'], morph)]

    for i in dataset[-10:]:
        print(i['ref'])
    for i in dataset[:10]:
        print(i['ref'])

    print(len(dataset))

    json.dump(dataset, open('unsup_dataset/unsup_dataset_transcripts_simple_morph.csv.json', 'w'), ensure_ascii = False)


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


if __name__ == '__main__':

    main()