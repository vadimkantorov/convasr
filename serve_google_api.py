# pip install google-cloud-speech
# https://github.com/googleapis/python-speech
# https://googleapis.dev/python/speech/latest/index.html
# https://grpc.io/docs/tutorials/basic/python/
# https://github.com/grpc/grpc/blob/master/examples/python/route_guide/route_guide_server.py

import argparse
import concurrent.futures
import grpc
import google.cloud.speech_v1.proto.cloud_speech_pb2 as pb2
import google.cloud.speech_v1.proto.cloud_speech_pb2_grpc as pb2_grpc
import torch
import audio
import transcripts
import transcribe


class SpeechServicerImpl(pb2_grpc.SpeechServicer):
	def __init__(self, device, labels, frontend, model, decoder):
		self.device = device
		self.labels = labels
		self.model = model
		self.frontend = frontend
		self.decoder = decoder

	def Recognize(self, req, ctx):
		assert req.config.encoding == pb2.RecognitionConfig.LINEAR16

		signal, sample_rate = audio.read_audio(None, raw_bytes = req.audio.content, raw_sample_rate = req.config.sample_rate_hertz, raw_num_channels = req.config.audio_channel_count, dtype = 'int16', sample_rate = self.frontend.sample_rate, mono = True)
		x = signal
		logits, olen = self.model(x.to(self.device))
		decoded = self.decoder.decode(logits, olen)
		ts = (x.shape[-1] / sample_rate) * torch.linspace(0, 1, steps = logits.shape[-1])

		transcript = self.labels.decode(decoded[0], ts)
		hyp = transcripts.join(hyp = transcript)

		mktime = lambda t: dict(seconds = int(t), nanos = int((t - int(t)) * 1e9))
		return pb2.RecognizeResponse(
			results = [
				dict(
					alternatives = [
						dict(
							transcript = hyp,
							confidence = 1.0,
							words = [
								dict(
									word = t['hyp'],
									start_time = mktime(t['begin']),
									end_time = mktime(t['end']),
									speaker_tag = 0
								) for t in transcript
							]
						)
					],
					channel_tag = 1
				)
			]
		)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', required = True)
	parser.add_argument('--model')
	parser.add_argument('--device', default = 'cuda', choices = ['cpu', 'cuda'])
	parser.add_argument('--decoder', choices = ['GreedyDecoder'], default = 'GreedyDecoder')
	parser.add_argument('--fp16', choices = ['O0', 'O1', 'O2', 'O3'], default = None)
	parser.add_argument('--endpoint', default = '127.0.0.1:50000')
	parser.add_argument('--num-workers', type = int, default = 10)
	args = parser.parse_args()

	service_impl = SpeechServicerImpl(args.device, *transcribe.setup(args))

	server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers = args.num_workers))
	pb2_grpc.add_SpeechServicer_to_server(service_impl, server)
	server.add_insecure_port(args.endpoint)

	print('Serving google-cloud-speech API @', args.endpoint)
	server.start()
	server.wait_for_termination()
