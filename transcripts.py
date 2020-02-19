import torch
import torch.nn.functional as F

# filter for display

# filter for dataset creation by min/max duration, cer, good first/last word alignment

# does not filter anything out, can only merge

def strip(transcript, keys = []):
	return [{k : v for k, v in t.items() if k not in keys} for t in transcript]

def segment(speech, sample_rate = 1):
	_notspeech_ = ~F.pad(speech, [1, 1])
	channel_i_channel_j = torch.cat([(speech & _notspeech_[..., :-2]).nonzero(), (speech & _notspeech_[..., 2:]).nonzero()], dim = -1)
	return [dict(begin = i / sample_rate, end = j / sample_rate, channel = channel) for channel, i, _, j in channel_i_channel_j.tolist()]

def resegment(segments, max_segment_seconds):
	def filter_words(ws, i, w, first, last):
		res = [(k, u) for k, u in enumerate(ws) if (first or i < 0 or ws[i]['j'] < u['i']) and (last or u['j'] < w['i'])]
		i, ws = zip(*res) if res else ([i], [])
		return i[-1], list(ws)
	
	for r, h in segments:
		k, ws = (0, r) if r else (1, h)
		last_flushed_ind = [-1, -1]
		for j, w in enumerate(ws):
			first, last = last_flushed_ind[k] == -1, j == len(ws) - 1
			if last or (w['end'] - ws[last_flushed_ind[k] + 1]['begin'] > max_segment_seconds):
				last_flushed_ind[0], r_ = filter_words(r, last_flushed_ind[0], w, first, last)
				last_flushed_ind[1], h_ = filter_words(h, last_flushed_ind[1], w, first, last)
				if r_ or h_:
					yield [r_, h_]

def summary(transcript, ij = False):
	res = dict(channel = list(set(t['channel'] for t in transcript))[0], begin = min(w['begin'] for w in transcript), end = max(w['end'] for w in transcript), i = min([w['i'] for w in transcript if 'i' in w] or [0]), j = max([w['j'] for w in transcript if 'j' in w] or [0])) if len(transcript) > 0 else dict(begin = 0, end = 0, i = 0, j = 0, channel = 0)
	if not ij:
		del res['i']
		del res['j']
	return res

def sort(transcript):
	return sorted(transcript, key = lambda t: sort_key(summary(t['alignment']['ref'] + t['alignment']['hyp'])))

def sort_key(t):
	return t.get('begin'), t.get('end'), t.get('channel')

def filter(transcript, min_cer = None, max_cer = None, min_duration = None, max_duration = None, time_gap = None, align_boundary_words = False):
	is_aligned = lambda w: w['type'] == 'ok'
	duration_check = lambda t: (min_duration is None or min_duration <= t['end'] - t['begin']) and (max_duration is None or t['end'] - t['begin'] <= max_duration)
	cer_check = lambda t: (min_cer is None or min_cer <= t['cer']) and (max_cer is None or t['cer'] <= max_cer)
	boundary_check = lambda t: ((not t['words']) or (not align_boundary_words) or (is_aligned(t['words'][0]) and is_aligned(t['words'][-1])))
	gap_check = lambda t, prev: time_gap is None or t['begin'] - prev['end'] >= time_gap
	
	prev = None
	for t in transcript:
		if duration_check(t) and cer_check(t) and boundary_check(t) and (prev is None or gap_check(t, prev)):
			prev = t
			yield t
