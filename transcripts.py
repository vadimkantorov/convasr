import os
import json
import torch
import torch.nn.functional as F

import audio
import utils

default_speaker_names = '_' + ''.join(chr(ord('A') + i) for i in range(26))
default_channel_names = {channel_missing : 'channel_', 0 : 'channel0', 1 : 'channel1'}

ref_missing = ''
speaker_name_missing = ''
speaker_missing = 0
channel_missing = -1
time_missing = -1
_er_missing = -1.0

def flatten(segments):
	return utils.flatten(segments)

def load(data_path):
	assert os.path.exists(data_path)
	
	if data_path.endswith('.rttm'):
		with open(file_path) as f:
			transcript = [dict(audio_name = splitted[1], begin = float(splitted[3]), end = float(splitted[3]) + float(splitted[4]), speaker_name = splitted[7]) for splitted in map(str.split, f)]
	
	elif data_path.endswith('.json') or data_path.endswith('.json.gz'):
		with utils.open_maybe_gz(data_path) as f:
			transcript = json.load(f)
	
	elif os.path.exists(data_path + '.json'):
		with open(data_path + '.json') as f: 
			transcript = json.load(f)
			for t in transcript:
				t['audio_path'] = data_path
	else:
		transcript = [dict(audio_path = data_path)]

	return transcript

def save(data_path, transcript):
	with open(data_path, 'w') as f:

		if data_path.endswith('.json'):
			json.dump(transcript, f, ensure_ascii = False, sort_keys = True, indent = 2)
		
		elif data_path.endswith('.rttm'):
			audio_name_ = audio_name(transcript[0])
			f.writelines('SPEAKER {audio_name} 1 {begin:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n'.format(audio_name = audio_name, begin = t['begin'], duration = compute_duration(t), speaker = t['speaker']) for t in transcript if t['speaker'] != speaker_missing)


def strip(transcript, keys = []):
	return [{k: v for k, v in t.items() if k not in keys} for t in transcript]


def join(ref = [], hyp = []):
	return ' '.join(filter(bool, [t.get('ref', '') for t in ref] + [t.get('hyp', '') for t in hyp])).strip()

def remap_speaker(transcript, speaker_perm):
	speaker_names = collect_speaker_names(transcript, num_speakers = len(speaker_perm) - 1)
	for t in transcript:
		speaker_ = speaker_perm[t['speaker']]
		t['speaker'], t['speaker_name'] = speaker_, speaker_names_[speaker_]


def collect_speaker_names(transcript, speaker_names = [], num_speakers = None, set_speaker = False):
	#TODO: convert channel to 0+
	
	if not transcript:
		return

	has_speaker = has_speaker_names = has_channel = True
	for t in transcript:
		has_speaker &= t.get('speaker') is not None 
		has_speaker_names &= bool(t.get('speaker_name'))
		has_channel &= t.get('channel', channel_missing) != channel_missing 

	if not speaker_names:
		if has_speaker_names:
			speaker_names = [speaker_name_missing] + sorted(set(t['speaker_name'] for t in transcript))
			speaker_names_index = {speaker_name_missing : speaker_missing, **{speaker_name : i for i, speaker_name in enumerate(speaker_names)}}

		elif has_channel:
			speaker_names_index = {default_channel_names[speaker] : speaker for speaker in [channel_missing] + list(range(num_speakers or 2))}
			speaker_names = [default_channel_names[channel] for channel in range(num_channels)]

		elif has_speaker:
			speaker_names = {t['speaker'] : default_speaker_names[t['speaker']] for t in transcript}
			assert speaker_missing not in speaker_names
			speaker_names[speaker_missing] = speaker_name_missing
			speaker_names = [speaker_names.get(speaker, speaker_name_missing) for speaker in range(1 + max(speaker_names.values()))]

	assert (not num_speakers) or len(speaker_names) >= 1 + num_speakers
	
	if set_speaker and (not has_speaker):
		for t in transcript:
			speaker_name = t.get('speaker_name', default_channel_names[t.get('channel', channel_missing)])
			t['speaker'] = speaker_names_index[speaker_name]

	return speaker_names

def speaker_name(ref = None, hyp = None):
	return ', '.join(sorted(filter(bool, set(t.get('speaker_name') for t in ref + hyp)))) or None

def segment_by_time(transcript, max_segment_seconds, break_on_speaker_change = True, break_on_channel_change = True):
	transcript = [t for t in transcript if t['begin'] != time_missing and t['end'] != time_missing]
	ind_last_taken = -1
	for j, t in enumerate(transcript):
		first, last = ind_last_taken == -1, j == len(transcript) - 1
		
		if last or (t['end'] - transcript[ind_last_taken + 1]['begin'] > max_segment_seconds) \
				or (break_on_speaker_change and j >= 1 and t['speaker'] != transcript[j - 1]['speaker']) \
				or (break_on_channel_change and j >= 1 and t['channel'] != transcript[j - 1]['channel']):

			ind_last_taken, transcript_segment = take_between(transcript, ind_last_taken, t, first, last, sort_by_time=False)
			if transcript_segment:
				yield transcript_segment

def take_between(transcript, ind_last_taken, t, first, last, sort_by_time = True, soft = True, set_speaker = False):
	if sort_by_time:
		lt = lambda a, b: a['end'] < b['begin']
		gt = lambda a, b: a['begin'] > b['begin']
	else:
		lt = lambda a, b: sort_key(a) < sort_key(b)
		gt = lambda a, b: sort_key(a) > sort_key(b)

	if soft:
		res = [(k, u) for k, u in enumerate(transcript)	if (first or ind_last_taken < 0 or lt(transcript[ind_last_taken], u)) and (last or gt(t, u))]
	else:
		intersects = lambda t, begin, end: (begin <= t['end'] and t['begin'] <= end)
		res = [(k, u) for k, u in enumerate(transcript) if ind_last_taken < k and intersects(t, u['begin'], u['end'])] if t else []

	ind_last_taken, transcript = zip(*res) if res else ([ind_last_taken], [])

	if set_speaker:
		for u in transcript:
			u['speaker'] = t.get('speaker', speaker_missing)
			if t.get('speaker_name') is not None:
				u['speaker_name'] = t['speaker_name']

	return ind_last_taken[-1], list(transcript)

def segment_by_ref(transcript, ref_segments, soft = True, set_speaker = set_speaker):
	ind_last_taken = -1
	if len(ref_segments) == 0:
		return []

	for j in range(len(ref_segments)):
		first, last = ind_last_taken == -1, j == len(ref_segments) - 1
		ind_last_taken, transcript_segment = take_between(transcript, ind_last_taken, summary(ref_segments[j]), first, last, sort_by_time=True, soft = soft, set_speaker = set_speaker)

		yield transcript_segment


def summary(transcript, ij = False):
	res = dict(
		begin = min(w.get('begin', 0.0) for w in transcript),
		end = max(w.get('end', 0.0) for w in transcript),
		i = min([w['i'] for w in transcript if 'i' in w] or [0]),
		j = max([w['j'] for w in transcript if 'j' in w] or [0])
	) if len(transcript) > 0 else dict(begin = time_missing, end = time_missing, i = 0, j = 0)
	if not ij:
		del res['i']
		del res['j']
	return res


def sort(transcript):
	return sorted(transcript, key = lambda t: sort_key(summary(t.get('words_ref', []) + t.get('words_hyp', []))))


def sort_key(t):
	return (t.get('audio_path'), t.get('begin'), t.get('end'), t.get('channel'))


def group_key(t):
	return t.get('audio_path')


def prune(
	transcript,
	align_boundary_words = False,
	cer = None,
	wer = None,
	mer = None,
	duration = None,
	gap = None,
	num_speakers = None,
	audio_name = None,
	unk = None,
	groups = None
):
	is_aligned = lambda w: (w.get('type') or w.get('error_tag')) == 'ok'
	duration_check = lambda t: duration is None or duration[0] <= compute_duration(t) <= duration[1]
	boundary_check = lambda t: ((not t.get('words')) or (not align_boundary_words) or
								(is_aligned(t['words'][0]) and is_aligned(t['words'][-1])))
	gap_check = lambda t, prev: prev is None or gap is None or gap[0] <= t['begin'] - prev['end'] <= gap[1]
	unk_check = lambda t: unk is None or unk[0] <= t.get('ref', '').count('*') <= unk[1]
	speakers_check = lambda t: num_speakers is None or num_speakers[0] <= (t.get('speaker') or ''
																			).count(',') + 1 <= num_speakers[1]
	cer_check = lambda t: cer is None or t.get('cer') is None or cer[0] <= t['cer'] <= cer[1]
	wer_check = lambda t: wer is None or t.get('wer') is None or wer[0] <= t['wer'] <= wer[1]
	mer_check = lambda t: mer is None or t.get('mer') is None or mer[0] <= t['mer'] <= mer[1]
	groups_check = lambda t: groups is None or t.get('group') is None or t['group'] in groups

	prev = None
	for t in transcript:
		if groups_check(t) and unk_check(t) and duration_check(t) and cer_check(t) and wer_check(t) and mer_check(
			t) and boundary_check(t) and gap_check(t, prev) and speakers_check(t):
			yield t
		prev = t


def compute_duration(t, hours = False):
	seconds = None
	
	if 'begin' in t or 'end' in t:
		seconds = t.get('end', 0) - t.get('begin', 0)
	elif 'hyp' in t or 'ref' in t:
		seconds = max(t_['end'] for k in ['hyp', 'ref'] for t_ in t.get(k, []))
	elif 'audio_path' in t:
		seconds = audio.compute_duration(t['audio_path'])

	assert seconds is not None

	return seconds / (60 * 60) if hours else seconds


def audio_name(t):
	return (t.get('audio_name') or os.path.basename(t['audio_path'])) if isinstance(t, dict) else os.path.basename(t)


number_tuple = lambda s: tuple(
	map(
		lambda ip: (float(ip[1]) if '.' in ip[1] else int(ip[1])) if ip[1] else float(['-inf', 'inf'][ip[0]]),
		enumerate((s if '-' in s else s + '-' + s).split('-'))
	)
)
