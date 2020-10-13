# pip install pyannote.metrics
# pip install pyannote.pipeline
# pip install pescador
# pip install optuna
# pip install filelock
# pip install git+https://github.com/pyannote/pyannote-audio
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import audio
import models
import shaping
import vis
import transcripts

import pyannote.core
import pyannote.database.util 
import pyannote.metrics.diarization
import webrtcvad

class WebrtcSpeechActivityDetectionModel(nn.Module):
	def __init__(self, aggressiveness):
		self.vad = webrtcvad.Vad(aggressiveness)
	
	def forward(self, signal, sample_rate, window_size = 0.02, extra = {}):
		assert sample_rate in [8_000, 16_000, 32_000, 48_000] and signal.dtype == torch.int16 and window_size in [0.01, 0.02, 0.03]
		frame_len = int(window_size * sample_rate)
		speech = torch.as_tensor([[len(chunk) == frame_len and self.vad.is_speech(bytearray(chunk.numpy()), sample_rate) for chunk in channel.split(frame_len)]	for channel in signal])
		transcript = [dict(begin = float(begin) * window_size, end = (float(begin) + float(duration)) * window_size, speaker = 1 + channel, speaker_name = transcripts.default_speaker_names[1 + channel], **extra) for channel in range(len(signal)) for begin, duration, mask in zip(*models.rle1d(speech[speaker])) if mask == 1]
		return transcript


class PyannoteDiarizationModel(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self.pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia', **kwargs)

	def forward(self, signal, sample_rate, extra = {}):
		#assert sample_rate == 16_000
		res = self.pipeline(dict(waveform = signal.t().numpy(), sample_rate = sample_rate))
		transcript = [dict(begin = turn.start, end = turn.end, speaker_name = speaker, **extra) for turn, _, speaker in res.itertracks(yield_label = True)]
		return transcript

def resize_to_min_size_(*tensors, dim = -1):
	size = min(t.shape[dim] for t in tensors)
	for t in tensors:
		if t.shape[dim] > size:
			sliced = t.narrow(dim, 0, size)
			t.set_(t.storage(), 0, sliced.size(), sliced.stride())

def convert_speaker_id(speaker_id, to_bipole = False, from_bipole = False):
	k, b = (1 - 3/2, 3 / 2) if from_bipole else (-2, 3) if to_bipole else (None, None)
	return (speaker_id != 0) * (speaker_id * k + b)

def select_speaker(signal : shaping.BT, kernel_size_smooth_silence : int, kernel_size_smooth_signal : int, kernel_size_smooth_speaker : int, silence_absolute_threshold : float = 0.2, silence_relative_threshold : float = 0.5, eps : float = 1e-9, normalization_percentile = 0.9) -> shaping.T:
	#TODO: remove bipole processing, smooth every speaker, conditioned on the other speaker

	assert len(signal) == 2

	padding = kernel_size_smooth_signal // 2
	stride = 1
	smoothed_for_diff = F.max_pool1d(signal.abs().unsqueeze(1), kernel_size_smooth_signal, stride = stride, padding = padding).squeeze(1)

	padding = kernel_size_smooth_silence // 2
	stride = 1
	
	# dilation
	smoothed_for_silence = F.max_pool1d(signal.abs().unsqueeze(1), kernel_size_smooth_silence, stride = stride, padding = padding).squeeze(1)
	
	# erosion
	smoothed_for_silence = -F.max_pool1d(-smoothed_for_silence.unsqueeze(1), kernel_size_smooth_silence, stride = stride, padding = padding).squeeze(1)
	
	# primitive VAD
	signal_max = smoothed_for_diff.kthvalue(int(normalization_percentile * smoothed_for_diff.shape[-1]), dim = -1, keepdim = True).values
	silence_absolute = smoothed_for_silence < silence_absolute_threshold
	silence_relative = smoothed_for_silence / (eps + signal_max) < silence_relative_threshold
	silence = silence_absolute | silence_relative
	
	diff_flat = smoothed_for_diff[0] - smoothed_for_diff[1]
	speaker_id_bipole = diff_flat.sign()
	
	padding = kernel_size_smooth_speaker // 2
	stride = 1
	speaker_id_bipole = F.avg_pool1d(speaker_id_bipole.view(1, 1, -1), kernel_size = kernel_size_smooth_speaker, stride = stride, padding = padding).view(-1).sign()

	# removing 1 sample silence at 1111-1-1-1-1 boundaries, replace by F.conv1d (patterns -101, 10-1)
	speaker_id_bipole = torch.where((speaker_id_bipole == 0) & (F.avg_pool1d(speaker_id_bipole.abs().view(1, 1, -1), kernel_size = 3, stride = 1, padding = 1).view(-1) == 2/3) & (F.avg_pool1d(speaker_id_bipole.view(1, 1, -1), kernel_size = 3, stride = 1, padding = 1).view(-1) == 0), torch.ones_like(speaker_id_bipole), speaker_id_bipole)
	
	resize_to_min_size_(silence, speaker_id_bipole, dim = -1)
	
	silence_flat = silence.all(dim = 0)
	speaker_id_categorical = convert_speaker_id(speaker_id_bipole, from_bipole = True) * (~silence_flat)

	bipole = torch.tensor([1, -1], dtype = speaker_id_bipole.dtype, device = speaker_id_bipole.device)
	speaker_id_mask = (~silence) * (speaker_id_bipole.unsqueeze(0) == bipole.unsqueeze(1))
	return speaker_id_categorical, torch.cat([silence_flat.unsqueeze(0), speaker_id_mask])

	#speaker_id = torch.where(silence.any(dim = 0), torch.tensor(0, device = signal.device, dtype = speaker_id.dtype), speaker_id)
	#return speaker_id, silence_flat, torch.stack((speaker_id * 0.5, diff, smoothed_for_silence[0], smoothed_for_silence[1] , silence_flat.float() * 0.5))

def ref(input_path, output_path, sample_rate, window_size, device, max_duration, debug_audio, html, ext):
	os.makedirs(output_path, exist_ok = True)
	audio_source = ([(input_path, audio_name) for audio_name in os.listdir(input_path)] if os.path.isdir(input_path) else [(os.path.dirname(input_path), os.path.basename(input_path))])
	for i, (input_path, audio_name) in enumerate(audio_source):
		print(i, '/', len(audio_source), audio_name)
		audio_path = os.path.join(input_path, audio_name)
		noextname = audio_name[:-len(ext)]
		transcript_path = os.path.join(output_path, noextname + '.json')
		rttm_path = os.path.join(output_path, noextname + '.rttm')

		signal, sample_rate = audio.read_audio(audio_path, sample_rate = sample_rate, mono = False, dtype = 'float32', duration = max_duration)

		speaker_id_ref, speaker_id_ref_ = select_speaker(signal.to(device), silence_absolute_threshold = 0.05, silence_relative_threshold = 0.2, kernel_size_smooth_signal = 128, kernel_size_smooth_speaker = 4096, kernel_size_smooth_silence = 4096)

		transcript = [dict(audio_path = audio_path, begin = float(begin) / sample_rate, end = (float(begin) + float(duration)) / sample_rate, speaker = speaker, speaker_name = transcripts.default_speaker_names[speaker]) for speaker in range(1, len(speaker_id_ref_)) for begin, duration, mask in zip(*models.rle1d(speaker_id_ref_[speaker])) if mask == 1]
		
		#transcript = [dict(audio_path = audio_path, begin = float(begin) / sample_rate, end = (float(begin) + float(duration)) / sample_rate, speaker_name = str(int(speaker)), speaker = int(speaker)) for begin, duration, speaker in zip(*models.rle1d(speaker_id_ref.cpu()))]

		transcript_without_speaker_missing = [t for t in transcript if t['speaker'] != transcripts.speaker_missing]
		transcripts.save(transcript_path, transcript_without_speaker_missing)
		print(transcript_path)

		transcripts.save(rttm_path, transcript_without_speaker_missing)
		print(rttm_path)
		
		if debug_audio:
			audio.write_audio(transcript_path + '.wav', torch.cat([signal[..., :speaker_id_ref.shape[-1]], convert_speaker_id(speaker_id_ref[..., :signal.shape[-1]], to_bipole = True).unsqueeze(0).cpu() * 0.5, speaker_id_ref_[..., :signal.shape[-1]].cpu() * 0.5]), sample_rate, mono = False)
			print(transcript_path + '.wav')

		if html:
			html_path = os.path.join(output_path, audio_name + '.html')
			vis.transcript(html_path, sample_rate = sample_rate, mono = True, transcript = transcript, duration = max_duration)

def hyp(input_path, output_path, device, batch_size, html, ext, sample_rate, max_duration):
	
	os.makedirs(output_path, exist_ok = True)
	audio_source = ([(input_path, audio_name) for audio_name in os.listdir(input_path)] if os.path.isdir(input_path) else [(os.path.dirname(input_path), os.path.basename(input_path))])
	model = PyannoteDiarizationModel(device = device, batch_size = batch_size)
	for i, (input_path, audio_name) in enumerate(audio_source):
		print(i, '/', len(audio_source), audio_name)
		audio_path = os.path.join(input_path, audio_name)
		noextname = audio_name[:-len(ext)]
		transcript_path = os.path.join(output_path, noextname + '.json')
		rttm_path = os.path.join(output_path, noextname + '.rttm')
	
		signal, sample_rate = audio.read_audio(audio_path, sample_rate = sample_rate, mono = True, dtype = 'float32', duration = max_duration)
		transcript = model(signal, sample_rate = sample_rate, extra = dict(audio_path = audio_path))
		transcripts.collect_speaker_names(transcript, set_speaker = True)
		
		transcripts.save(transcript_path, transcript)
		print(transcript_path)

		transcripts.save(rttm_path, transcript)
		print(rttm_path)

		if html:
			html_path = os.path.join(output_path, audio_name + '.html')
			vis.transcript(html_path, sample_rate = sample_rate, mono = True, transcript = transcript, duration = max_duration)

def der(ref_rttm_path, hyp_rttm_path, metric = pyannote.metrics.diarization.DiarizationErrorRate()):
	ref, hyp = map(pyannote.database.util.load_rttm, [ref_rttm_path, hyp_rttm_path])
	ref, hyp = [next(iter(anno.values())) for anno in [ref, hyp]]
	return metric(ref, hyp)

def speaker_mask(transcript, num_speakers, duration, sample_rate):
	mask = torch.zeros(1 + num_speakers, int(duration * sample_rate), dtype = torch.bool)
	for t in transcript:
		mask[t['speaker'], int(t['begin'] * sample_rate) : int(t['end'] * sample_rate)] = 1
	mask[0] = mask[1] & mask[2]
	return mask

def speaker_error(ref, hyp, num_speakers, sample_rate = 8000, hyp_speaker_mapping = None, ignore_silence_and_overlapped_speech = True):
	assert num_speakers == 2
	duration = transcripts.compute_duration(dict(ref = ref, hyp = hyp))
	ref_mask = speaker_mask(ref,  num_speakers, duration, sample_rate)
	hyp_mask_ = speaker_mask(hyp, num_speakers, duration, sample_rate)

	print('duration', duration)
	vals = []
	for hyp_perm in ([[0, 1, 2], [0, 2, 1]] if hyp_speaker_mapping is None else hyp_speaker_mapping):
		hyp_mask = hyp_mask_[hyp_perm]
		speaker_mismatch = (ref_mask[1] != hyp_mask[1]) | (ref_mask[2] != hyp_mask[2])
		if ignore_silence_and_overlapped_speech:
			silence_or_overlap_mask = ref_mask[1] == ref_mask[2]
			speaker_mismatch = speaker_mismatch[~silence_or_overlap_mask]

		confusion = (hyp_mask[1] & ref_mask[2] & (~ref_mask[1])) | (hyp_mask[2] & ref_mask[1] & (~ref_mask[2]))
		false_alarm = (hyp_mask[1] | hyp_mask[2]) & (~ref_mask[1]) & (~ref_mask[2])
		miss = (~hyp_mask[1]) & (~hyp_mask[2]) & (ref_mask[1] | ref_mask[2])
		total = ref_mask[1] | ref_mask[2]

		confusion, false_alarm, miss, total = [float(x.float().mean()) * duration for x in [confusion, false_alarm, miss, total]]

		print('my', 'confusion', confusion, 'false_alarm', false_alarm, 'miss', miss, 'total', total)
		err = float(speaker_mismatch.float().mean())
		vals.append((err, hyp_perm))

	return min(vals)
		
def eval(ref, hyp, html, debug_audio, sample_rate = 100):
	if os.path.isfile(ref) and os.path.isfile(hyp):
		print(der(ref_rttm_path = ref, hyp_rttm_path = hyp))

	elif os.path.isdir(ref) and os.path.isdir(hyp):
		errs = []
		diarization_transcript = []
		for rttm in os.listdir(ref):
			if not rttm.endswith('.rttm'):
				continue

			print(rttm)
			audio_path = transcripts.load(os.path.join(hyp, rttm).replace('.rttm', '.json'))[0]['audio_path']

			ref_rttm_path, hyp_rttm_path = os.path.join(ref, rttm), os.path.join(hyp, rttm)
			ref_transcript, hyp_transcript = map(transcripts.load, [ref_rttm_path, hyp_rttm_path])
			ser_err, hyp_perm = speaker_error(ref = ref_transcript, hyp = hyp_transcript, num_speakers = 2, sample_rate = sample_rate, ignore_silence_and_overlapped_speech = True)
			der_err, *_ = speaker_error(ref = ref_transcript, hyp = hyp_transcript, num_speakers = 2, sample_rate = sample_rate, ignore_silence_and_overlapped_speech = False)
			der_err_ = der(ref_rttm_path = ref_rttm_path, hyp_rttm_path = hyp_rttm_path)
			transcripts.remap_speaker(hyp_transcript, hyp_perm)

			err = dict(
				ser = ser_err,
				der = der_err,
				der_ = der_err_
			)
			diarization_transcript.append(dict(
				audio_path = audio_path,
				audio_name = transcripts.audio_name(audio_path),
				ref = ref_transcript, 
				hyp = hyp_transcript,
				**err
			))
			print(rttm, '{ser:.2f}, {der:.2f} | {der_:.2f}'.format(**err))
			print()
			errs.append(err)
		print('===')
		print({k : sum(e) / len(e) for k in errs[0] for e in [[err[k] for err in errs]]})
		
		if html:
			print(vis.diarization(sorted(diarization_transcript, key = lambda t: t['ser'], reverse = True), html, debug_audio))


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
	cmd.add_argument('--ext', default = '.mp3')
	cmd.set_defaults(func = ref)
	
	cmd = subparsers.add_parser('hyp')
	cmd.add_argument('--device', default = 'cuda')
	cmd.add_argument('--input-path', '-i')
	cmd.add_argument('--output-path', '-o')
	cmd.add_argument('--batch-size', type = int, default = 8)
	cmd.add_argument('--sample-rate', type = int, default = 16_000)
	cmd.add_argument('--html', action = 'store_true')
	cmd.add_argument('--ext', default = '.mp3.wav')
	cmd.add_argument('--max-duration', type = float)
	cmd.set_defaults(func = hyp)
	
	cmd = subparsers.add_parser('eval')
	cmd.add_argument('--ref', required = True)
	cmd.add_argument('--hyp', required = True)
	cmd.add_argument('--html', default = 'data/diarization.html')
	cmd.add_argument('--audio', dest = 'debug_audio', action = 'store_true')
	cmd.set_defaults(func = eval)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)

