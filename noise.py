import os
import numpy as np
import argparse
import tempfile
import torch
import librosa
import dataset

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-path')
parser.add_argument('-n', '--noise-path')
parser.add_argument('-o', '--output-path', default='test.wav')
parser.add_argument('--sample-rate', type = int, default=16000)
parser.add_argument('--noise-level', type=float, default=1.0)
args = parser.parse_args()

data, sample_rate = dataset.read_wav(args.input_path)
data = torch.from_numpy(librosa.resample(data.numpy(), sample_rate, args.sample_rate))

noise, sample_rate = dataset.read_wav(args.noise_path)
noise = torch.from_numpy(librosa.resample(noise.numpy(), sample_rate, args.sample_rate))
#silence = noise.abs() < args.silence_level


noise = torch.cat([noise] * (1 + len(data) // len(noise)))[:len(data)]

for k in range(10):
    noise_level = k / 10.
    mixed = data + noise_level * noise
    librosa.output.write_wav(args.output_path + f'{k}.wav', mixed.numpy() / float(mixed.abs().max()), args.sample_rate)
