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
import transcribe

class SpeechServicerImpl(pb2_grpc.SpeechServicer):
	def __init__(self, labels, model, decoder):
		self.labels = labels
		self.model = model
		self.decoder = decoder

	def Recognize(self, request, context):
		config = request.config
		audio = request.audio

		return pb2.RecognizeResponse(results = [
			pb2.SpeechRecognitionResult(
				alternatives = pb2.SpeechRecognitionAlternative(
					transcript = 'transcript', 
					confidence = 1.0, 
					words = [
						pb2.WordInfo(word = str(k), start_time = '1.2s', end_time = '2.3s', speaker_tag = 31337) for k in range(3)
					]
				),
				channel = 0
			)
		])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', required = True)
	parser.add_argument('--model')
	parser.add_argument('--device', default = 'cuda', choices = ['cpu', 'cuda'])
	parser.add_argument('--decoder', choices = ['GreedyDecoder'], default = 'GreedyDecoder')
	parser.add_argument('--fp16', choices = ['O0', 'O1', 'O2', 'O3'], default = None)
	parser.add_argument('--endpoint', default = 'localhost' + ':' + str(50051))
	parser.add_argument('--num-workers', type = int, default = 10)
	args = parser.parse_args()
	
	service_impl = SpeechServicerImpl(*transcribe.setup(args))

	server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers = args.num_workers))
	pb2_grpc.add_SpeechServicer_to_server(service_impl, server)
	server.add_insecure_port(args.endpoint)
	
	print('Serving google-cloud-speech API @', args.endpoint)
	server.start()
	server.wait_for_termination()
