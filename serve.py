# pip install google-cloud-speech
# https://github.com/googleapis/python-speech
# https://googleapis.dev/python/speech/latest/index.html
# https://grpc.io/docs/tutorials/basic/python/
# https://github.com/grpc/grpc/blob/master/examples/python/route_guide/route_guide_server.py

import argparse
import concurrent
import grpc
import google.cloud.speech_v1
import google.cloud.speech_v1.proto.cloud_speech_pb2
import google.cloud.speech_v1.proto.cloud_speech_pb2_grpc
import transcribe

def Recognize(self, request, context):
	pass

def LongRunningRecognize(self, request, context):
	pass

def StreamingRecognize(self, request_iterator, context):
	pass

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', required = True)
	parser.add_argument('--model')
	parser.add_argument('--device', default = 'cuda', choices = ['cpu', 'cuda'])
	parser.add_argument('--decoder', choices = ['GreedyDecoder'], default = 'GreedyDecoder')
	parser.add_argument('--fp16', choices = ['O0', 'O1', 'O2', 'O3'], default = None)
	parser.add_argument('--port', type = int, default = 50051)
	parser.add_arugment('--num-workers', type = int, default = 10)
	args = parser.parse_args()
	
	labels, model, decoder = transcribe.setup(args)
	
	impl = type('SpeechServicerImpl', (google.cloud.speech_v1.proto.cloud_speech_pb2_grpc.SpeechServicer,), dict(Recognize = Recognize, LongRunningRecognize = LongRunningRecognize, StreamingRecognize = StreamingRecognize))()

	server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers = args.num_workers))
	google.cloud.speech_v1.proto.cloud_speech_pb2_grpc.add_SpeechServicer_to_server(impl, server)
	server.add_insecure_port(f'[::]:{args.port}')
	server.start()
	server.wait_for_termination()
