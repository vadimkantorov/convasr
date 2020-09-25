import os
import argparse
import json
import torch
import torch.nn.functional as F
import audio
import models
import shaping
import vis

def resize_to_min_size_(*tensors, dim = -1):
	size = min(t.shape[dim] for t in tensors)
	for t in tensors:
		if t.shape[dim] > size:
			sliced = t.narrow(dim, 0, size)
			t.set_(t.storage(), 0, sliced.size(), sliced.stride())


def convert_speaker_id(speaker_id, to_bipole = False, from_bipole = False):
	k, b = (1 - 3/2, 3 / 2) if from_bipole else (-2, 3) if to_bipole else (None, None)
	return (speaker_id != 0) * (speaker_id * k + b)

def select_speaker(signal : shaping.BT, kernel_size_smooth_silence : int, kernel_size_smooth_signal : int, kernel_size_smooth_speaker : int, silence_absolute_threshold : float = 0.2, silence_relative_threshold : float = 0.5, eps : float = 1e-9) -> shaping.T:
	assert len(signal) == 2

	padding = kernel_size_smooth_signal // 2
	stride = 1
	smoothed = F.max_pool1d(signal.abs().unsqueeze(1), kernel_size_smooth_signal, stride = stride, padding = padding).squeeze(1)

	padding = kernel_size_smooth_silence // 2
	stride = 1
	smoothed_for_silence = F.max_pool1d(signal.abs().unsqueeze(1), kernel_size_smooth_silence, stride = stride, padding = padding).squeeze(1)
	smoothed_for_silence = -F.max_pool1d(-smoothed_for_silence.unsqueeze(1), kernel_size_smooth_silence, stride = stride, padding = padding).squeeze(1)
	
	silence_absolute = smoothed_for_silence < silence_absolute_threshold
	#silence_relative = smoothed / (eps + smoothed.max(dim = -1, keepdim = True).values) < silence_relative_threshold
	#silence = silence_absolute | silence_relative
	silence = silence_absolute
	
	diff = smoothed[0] - smoothed[1]
	speaker_id = diff.sign()
	
	padding = kernel_size_smooth_speaker // 2
	stride = 1
	speaker_id = F.avg_pool1d(speaker_id.view(1, 1, -1), kernel_size = kernel_size_smooth_speaker, stride = stride, padding = padding).view(-1).sign()

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
		transcript_path = os.path.join(output_path, audio_name + '.json')

		signal, sample_rate = audio.read_audio(audio_path.replace('mono', 'stereo').replace('.mp3.wav', '.mp3'), sample_rate = sample_rate, mono = False, dtype = 'float32', duration = max_duration)

		speaker_id_ref, speaker_id_ref_ = select_speaker(signal.to(device), silence_absolute_threshold = 0.2, silence_relative_threshold = 0.0, kernel_size_smooth_signal = 128, kernel_size_smooth_speaker = 4096, kernel_size_smooth_silence = 4096)
		
		transcript_ref = [dict(audio_path = audio_path, begin = float(begin) / sample_rate, end = (float(begin) + float(duration)) / sample_rate, speaker_name = str(int(speaker)), speaker = int(speaker)) for begin, duration, speaker in zip(*models.rle1d(speaker_id_ref.cpu()))]

		json.dump(transcript_ref, open(transcript_path, 'w'), indent = 2, sort_keys = True)
		print(transcript_path)
		
		if debug_audio:
			audio.write_audio(transcript_path + '.wav', torch.cat([signal[..., :speaker_id_ref.shape[-1]], convert_speaker_id(speaker_id_ref[..., :signal.shape[-1]], to_bipole = True).unsqueeze(0).cpu() * 0.5, speaker_id_ref_[..., :signal.shape[-1]].cpu() * 0.5]), sample_rate, mono = False)
			print(transcript_path + '.wav')

		if html:
			html_path = os.path.join(output_path, audio_name + '.html')
			vis.transcript(html_path, sample_rate = sample_rate, mono = True, transcript = transcript_path, duration = max_duration)

def hyp():
	pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia', device = device)#, batch_size = 8)
		#diarization = pipeline(dict(audio = audio_path))
		#transcript = [dict(audio_path = audio_path, begin = turn.start, end = turn.end, speaker_name = speaker) for turn, _, speaker in diarization.itertracks(yield_label = True)]
		#speaker_names = [None] + list(set(t['speaker_name'] for t in transcript))
		#for t in transcript:
		#	t['speaker'] = speaker_names.index(t['speaker_name'])
		#json.dump(transcript, open(transcript_path, 'w'), indent = 2, sort_keys = True)

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
	cmd.add_argument('--debug-audio', action = 'store_true')
	cmd.add_argument('--html', action = 'store_true')
	cmd.set_defaults(func = ref)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)

