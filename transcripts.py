import torch
import torch.nn.functional as F

# filter for display

# filter for dataset creation by min/max duration, cer, good first/last word alignment

# does not filter anything out, can only merge

def strip(transcript, keys = []):
	return [{k : v for k, v in t.items() if k not in keys} for t in transcript]

#def segment(speech, sample_rate = 1):
#	_notspeech_ = ~F.pad(speech, [1, 1])
#	channel_i_channel_j = torch.cat([(speech & _notspeech_[..., :-2]).nonzero(), (speech & _notspeech_[..., 2:]).nonzero()], dim = -1)
#	return [dict(begin = i / sample_rate, end = j / sample_rate, channel = channel) for channel, i, _, j in channel_i_channel_j.tolist()]

def join(ref = [], hyp = []):
	return ' '.join(t['ref'] for t in ref).strip() + ' '.join(t['hyp'] for t in hyp).strip()

def speaker(ref = None, hyp = None):
	return ', '.join(sorted(set(t['speaker'] or 'NA' for t in (ref if ref is not None else hyp if hyp is not None else []))))

def take_between(transcript, ind_last_taken, t, first, last):
	res = [(k, u) for k, u in enumerate(transcript) if (first or ind_last_taken < 0 or transcript[ind_last_taken]['end'] < u['begin']) and (last or u['end'] < t['begin'])]
	ind_last_taken, transcript = zip(*res) if res else ([ind_last_taken], [])
	return ind_last_taken[-1], list(transcript)

def segment(transcript, max_segment_seconds):
	ind_last_taken = -1
	if isinstance(max_segment_seconds, list):
		for j in range(len(max_segment_seconds)):
			first, last = ind_last_taken == -1, j == len(max_segment_seconds) - 1
			ind_last_taken, transcript_segment = take_between(transcript, ind_last_taken, max_segment_seconds[j + 1][0] if not last else None, first, last)
			yield transcript_segment
	else:
		for j, t in enumerate(transcript):
			first, last = ind_last_taken == -1, j == len(transcript) - 1
			if last or (t['end'] - transcript[ind_last_taken + 1]['begin'] > max_segment_seconds) or t['speaker'] != transcript[ind_last_taken + 1]['speaker']:
				ind_last_taken, transcript_segment = take_between(transcript, ind_last_taken, t, first, last)
				if transcript_segment:
					yield transcript_segment
	
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

float_tuple = lambda s: tuple(map(lambda ip: float(ip[1] if ip[1] else ['-inf', 'inf'][ip[0]]) , enumerate((s if '-' in s else s + '-' + s).split('-'))))

def filter(transcript, align_boundary_words = False, cer = None, wer = None, duration = None, gap = None, num_speakers = None, audio_name = None):
	is_aligned = lambda w: w['type'] == 'ok'
	duration_check = lambda t: duration is None or duration[0] <= t['end'] - t['begin'] <= duration[1]
	cer_check = lambda t: cer is None or cer[0] <= t['cer'] <= cer[1]
	boundary_check = lambda t: ((not t['words']) or (not align_boundary_words) or (is_aligned(t['words'][0]) and is_aligned(t['words'][-1])))
	gap_check = lambda t, prev: prev is None or gap is None or gap[0] <= t['begin'] - prev['end'] <= gap[1]
	speakers_check = lambda t: num_speakers is None or num_speakers[0] <= t.get('speaker', '').count(',') + 1 <= num_speakers[1]

	prev = None
	for t in transcript:
		if duration_check(t) and cer_check(t) and boundary_check(t) and gap_check(t, prev) and speakers_check(t):
			prev = t
			yield t
