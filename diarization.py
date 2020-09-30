import os
import argparse
import json
import torch
import torch.nn.functional as F
import audio
import models
import shaping
import vis
import transcripts

import pyannote.core
import pyannote.database.util 
import pyannote.metrics.diarization

def resize_to_min_size_(*tensors, dim = -1):
	size = min(t.shape[dim] for t in tensors)
	for t in tensors:
		if t.shape[dim] > size:
			sliced = t.narrow(dim, 0, size)
			t.set_(t.storage(), 0, sliced.size(), sliced.stride())

def convert_speaker_id(speaker_id, to_bipole = False, from_bipole = False):
	k, b = (1 - 3/2, 3 / 2) if from_bipole else (-2, 3) if to_bipole else (None, None)
	return (speaker_id != 0) * (speaker_id * k + b)

def write_rttm(file_path, transcript):
	audio_name = transcripts.audio_name(transcript[0])
	with open(file_path, 'w') as f:
		f.writelines('SPEAKER {audio_name} 1 {begin:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n'.format(audio_name = audio_name, begin = t['begin'], duration = transcripts.compute_duration(t), speaker = t['speaker']) for t in transcript)

def select_speaker(signal : shaping.BT, kernel_size_smooth_silence : int, kernel_size_smooth_signal : int, kernel_size_smooth_speaker : int, silence_absolute_threshold : float = 0.2, silence_relative_threshold : float = 0.5, eps : float = 1e-9, normalization_percentile = 0.9) -> shaping.T:
	assert len(signal) == 2

	padding = kernel_size_smooth_signal // 2
	stride = 1
	smoothed_for_diff = F.max_pool1d(signal.abs().unsqueeze(1), kernel_size_smooth_signal, stride = stride, padding = padding).squeeze(1)

	padding = kernel_size_smooth_silence // 2
	stride = 1
	smoothed_for_silence = F.max_pool1d(signal.abs().unsqueeze(1), kernel_size_smooth_silence, stride = stride, padding = padding).squeeze(1)
	smoothed_for_silence = -F.max_pool1d(-smoothed_for_silence.unsqueeze(1), kernel_size_smooth_silence, stride = stride, padding = padding).squeeze(1)
	
	signal_max = smoothed_for_diff.kthvalue(int(normalization_percentile * smoothed_for_diff.shape[-1]), dim = -1, keepdim = True).values
	silence_absolute = smoothed_for_silence < silence_absolute_threshold
	silence_relative = smoothed_for_silence / (eps + signal_max) < silence_relative_threshold
	silence = silence_absolute | silence_relative
	
	diff_flat = smoothed_for_diff[0] - smoothed_for_diff[1]
	speaker_id = diff_flat.sign()
	
	padding = kernel_size_smooth_speaker // 2
	stride = 1
	#TODO: remove 1 sample silence
	speaker_id = F.avg_pool1d(speaker_id.view(1, 1, -1), kernel_size = kernel_size_smooth_speaker, stride = stride, padding = padding).view(-1).sign()

	replace = (speaker_id == 0) & (F.avg_pool1d(speaker_id.abs().view(1, 1, -1), kernel_size = 3, stride = 1, padding = 1).view(-1) == 2/3) & (F.avg_pool1d(speaker_id.view(1, 1, -1), kernel_size = 3, stride = 1, padding = 1).view(-1) == 0)
	speaker_id = torch.where(replace, torch.ones_like(speaker_id), speaker_id)
	
	resize_to_min_size_(silence, speaker_id, dim = -1)
	
	silence_flat = silence.all(dim = 0)
	speaker_id_flat = convert_speaker_id(speaker_id, from_bipole = True) * (~silence_flat)

	speaker_id = (~silence) * (speaker_id.unsqueeze(0) == torch.tensor([1, -1], dtype = speaker_id.dtype, device = speaker_id.device).unsqueeze(1))
	return speaker_id_flat, torch.cat([silence_flat.unsqueeze(0), speaker_id])

	#speaker_id = torch.where(silence.any(dim = 0), torch.tensor(0, device = signal.device, dtype = speaker_id.dtype), speaker_id)
	#return speaker_id, silence_flat, torch.stack((speaker_id * 0.5, diff, smoothed_for_silence[0], smoothed_for_silence[1] , silence_flat.float() * 0.5))

def ref(input_path, output_path, sample_rate, window_size, device, max_duration, debug_audio, html):
	os.makedirs(output_path, exist_ok = True)
	audio_source = ([(input_path, audio_name) for audio_name in os.listdir(input_path)] if os.path.isdir(input_path) else [(os.path.dirname(input_path), os.path.basename(input_path))])
	for i, (input_path, audio_name) in enumerate(audio_source):
		print(i, '/', len(audio_source), audio_name)
		audio_path = os.path.join(input_path, audio_name)
		transcript_path = os.path.join(output_path, audio_name + '.ref.json')
		rttm_path = os.path.join(output_path, audio_name + '.ref.rttm')

		signal, sample_rate = audio.read_audio(audio_path, sample_rate = sample_rate, mono = False, dtype = 'float32', duration = max_duration)

		speaker_id_ref, speaker_id_ref_ = select_speaker(signal.to(device), silence_absolute_threshold = 0.05, silence_relative_threshold = 0.2, kernel_size_smooth_signal = 128, kernel_size_smooth_speaker = 4096, kernel_size_smooth_silence = 4096)
		
		transcript = [dict(audio_path = audio_path, begin = float(begin) / sample_rate, end = (float(begin) + float(duration)) / sample_rate, speaker_name = str(int(speaker)), speaker = int(speaker)) for begin, duration, speaker in zip(*models.rle1d(speaker_id_ref.cpu()))]

		json.dump([t for t in transcript if t['speaker'] != 0], open(transcript_path, 'w'), indent = 2, sort_keys = True)
		print(transcript_path)

		write_rttm(rttm_path, transcript)
		print(rttm_path)
		
		if debug_audio:
			audio.write_audio(transcript_path + '.wav', torch.cat([signal[..., :speaker_id_ref.shape[-1]], convert_speaker_id(speaker_id_ref[..., :signal.shape[-1]], to_bipole = True).unsqueeze(0).cpu() * 0.5, speaker_id_ref_[..., :signal.shape[-1]].cpu() * 0.5]), sample_rate, mono = False)
			print(transcript_path + '.wav')

		if html:
			html_path = os.path.join(output_path, audio_name + '.html')
			vis.transcript(html_path, sample_rate = sample_rate, mono = True, transcript = transcript, duration = max_duration)

def hyp(input_path, output_path, device, batch_size, html):
	pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia', device = device, batch_size = batch_size)
	
	os.makedirs(output_path, exist_ok = True)
	audio_source = ([(input_path, audio_name) for audio_name in os.listdir(input_path)] if os.path.isdir(input_path) else [(os.path.dirname(input_path), os.path.basename(input_path))])
	for i, (input_path, audio_name) in enumerate(audio_source):
		print(i, '/', len(audio_source), audio_name)
		audio_path = os.path.join(input_path, audio_name)
		transcript_path = os.path.join(output_path, audio_name + '.hyp.json')
		rttm_path = os.path.join(output_path, audio_name + '.hyp.rttm')
	
		res = pipeline(dict(audio = audio_path))
		transcript = [dict(audio_path = audio_path, begin = turn.start, end = turn.end, speaker_name = speaker) for turn, _, speaker in res.itertracks(yield_label = True)]
		speaker_names = transcripts.speaker_names(transcript)
		for t in transcript:
			t['speaker'] = speaker_names.index(t['speaker_name'])
		
		json.dump(transcript, open(transcript_path, 'w'), indent = 2, sort_keys = True)
		print(transcript_path)

		res.write_rttm(open(rttm_path, 'w'))
		print(rttm_path)

		if html:
			html_path = os.path.join(output_path, audio_name + '.html')
			vis.transcript(html_path, sample_rate = sample_rate, mono = True, transcript = transcript, duration = max_duration)
		
def der(ref, hyp):
	metric = pyannote.metrics.diarization.DiarizationErrorRate()
	ref, hyp = map(pyannote.database.util.load_rttm, [ref, hyp])
	ref, hyp = [next(iter(anno.values())) for anno in [ref, hyp]]
	der = metric(ref, hyp)
	print(der)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()
	
	cmd = subparsers.add_parser('ref')
	cmd.add_argument('--input-path', '-i')
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--sample-rate', type = int, default = 8_000)
	cmd.add_argument('--window-size', type = float, default = 0.02)
	cmd.add_argument('--device', default = 'cuda')
	cmd.add_argument('--max-duration', type = float)
	cmd.add_argument('--audio', dest = 'debug_audio', action = 'store_true')
	cmd.add_argument('--html', action = 'store_true')
	cmd.set_defaults(func = ref)
	
	cmd = subparsers.add_parser('hyp')
	cmd.add_argument('--device', default = 'cuda')
	cmd.add_argument('--input-path', '-i')
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--batch-size', type = int, default = 8)
	cmd.add_argument('--html', action = 'store_true')
	cmd.set_defaults(func = hyp)
	
	cmd = subparsers.add_parser('der')
	cmd.add_argument('--ref', required = True)
	cmd.add_argument('--hyp', required = True)
	cmd.set_defaults(func = der)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)

