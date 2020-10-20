import pickle

import arpa
import datasets
import labels
from old_ctc_beam_search import decode, timing


@timing
def time_arpa_p(lm):
	for line in ['да', 'да да', 'нет'] * 10:
		retrieve(line, lm)


@timing
def retrieve(line, lm):
	lm.log_p(line)


def main():
	with open('./data/to_decode.pkl', 'rb') as file:
		meta = pickle.load(file)

	lm_path = './data/chats_04_prune.arpa'
	lm_path = './data/chats_06_noprune_char.arpa'
	lm = arpa.loadf(lm_path)[0]
	lm.log_p('да')

	l = labels.Labels(labels.Language('ru'), name='char')

	to_decode = meta['log_probs'].squeeze(0).cpu().numpy()

	decoded, score, beam = decode(to_decode, blank=l.chr2idx['|'], lm=lm, beam_size=10, labels=labels, min_cutoff=1)
	decoded = [list(decoded)]
	other_beams = [list(e[0]) for e in beam]

	print(decoded, score)

	hyp_segments = [
		l.decode(
				decoded[i],
				None,
				channel=0,
				replace_blank=True,
				replace_blank_series=True,
				replace_repeat=True,
				replace_space=False,
				speaker=None
		) for i in range(len(decoded))
	]

	other_hyp_segments = [
		l.decode(
				other_beams[i],
				None,
				channel=0,
				replace_blank=True,
				replace_blank_series=True,
				replace_repeat=True,
				replace_space=False,
				speaker=None
		) for i in range(len(other_beams))
	]

	print('best: ', hyp_segments)

	for e in other_hyp_segments:
		print(e)

	for e in other_beams:
		print(e)


if __name__ == '__main__':
    main()