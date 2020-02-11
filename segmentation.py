import torch
import torch.nn.functional as F

# filter for display

# filter for dataset creation by min/max duration, cer, good first/last word alignment

# does not filter anything out, can only merge
def segment(speech, sample_rate = 1):
	_notspeech_ = ~F.pad(speech, [1, 1])
	(begin,), (end,) = (speech & _notspeech_[:-2]).nonzero(as_tuple = True), (speech & _notspeech_[2:]).nonzero(as_tuple = True)
	
	#sec = lambda k: k / len(idx) * (e - b)
	#i = 0
	#for j in range(1, 1 + len(idx)):
	#	if j == len(idx) or (idx[j] == labels.space_idx and sec(j - 1) - sec(i) > max_segment_seconds):
	#		yield (b + sec(i), b + sec(j - 1), labels.postprocess_transcript(labels.decode(idx[i:j])[0]))
	#		i = j + 1
	return [dict(i = i, j = j, begin = i / sample_rate, end = j / sample_rate) for i, j in zip(begin.tolist(), end.tolist())]

	#begin_end_ = ((frame_len * torch.IntTensor(begin_end)).float() / sample_rate).tolist()

def resegment(r, h, ws, max_segment_seconds):
	def filter_words(ws, i, w, first, last):
		res = [(k, u) for k, u in enumerate(ws) if (first or i is None or ws[i]['j'] < u['i']) and (last or u['j'] < w['i'])]
		if not res:
			return i, []
		i, ws = zip(*res)
		return i[-1], list(ws)
	
	k = [r, h].index(ws)
	last_flushed_ind = [-1, -1]
	for j, w in enumerate(ws):
		first_last = dict(first = last_flushed_ind[k] == -1, last = j == len(ws) - 1)
		if first_last['last'] or w['end'] - ws[last_flushed_ind[k] + 1]['begin'] > max_segment_seconds:
			last_flushed_ind[0], r_ = filter_words(r, last_flushed_ind[0], w, **first_last)
			last_flushed_ind[1], h_ = filter_words(h, last_flushed_ind[1], w, **first_last)
			yield [r_, h_]

def summary(ws):
	return dict(begin = min(w['begin'] for w in ws), end = max(w['end'] for w in ws), i = min(w['i'] for w in ws), j = max(w['j'] for w in ws)) if len(ws) > 0 else dict(begin = 0, end = 0, i = 0, j = 0)

def sort(segments):
	return list(sorted(segments, key = lambda s: tuple(map(summary(s[-1] + s[-2]).get, ['begin', 'end', 'channel']))))
