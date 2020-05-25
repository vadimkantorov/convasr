# pip install google-cloud-speech
# https://github.com/googleapis/python-speech
# https://googleapis.dev/python/speech/latest/index.html
# https://grpc.io/docs/tutorials/basic/python/
# https://github.com/grpc/grpc/blob/master/examples/python/route_guide/route_guide_server.py

import argparse
import concurrent.futures
import grpc
import google.cloud.speech_v1.proto.cloud_speech_pb2_grpc as pb2_grpc
import transcribe

class SpeechServicerImpl(pb2_grpc.SpeechServicer):
	def __init__(self, labels, model, decoder):
		self.labels = labels
		self.model = model
		self.decoder = decoder

	def Recognize(self, request, context):
		config = request.config
		audio = request.audio

		return dict(results = [
			dict(
				alternatives = [dict(
					transcript = 'transcript', 
					confidence = 1.0, 
					words = [
						dict(word = str(k), start_time = dict(seconds = 1, nanos = 100), end_time = dict(seconds = 2, nanos = 7), speaker_tag = 31337) for k in range(3)
					]
				)],
				channel_tag = 1
			)
		])

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
	
	service_impl = SpeechServicerImpl(*transcribe.setup(args))

	server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers = args.num_workers))
	pb2_grpc.add_SpeechServicer_to_server(service_impl, server)
	server.add_insecure_port(args.endpoint)
	
	print('Serving google-cloud-speech API @', args.endpoint)
	server.start()
	server.wait_for_termination()
