import os
import numpy as np
import argparse
import tempfile
import torch
import librosa
import dataset


parser = argparse.ArgumentParser()
parser.add_argument('--input-path', default='input.wav', help='The input audio to inject noise into')
parser.add_argument('--noise-path', default='noise.wav', help='The noise file to mix in')
parser.add_argument('--output-path', default='output.wav', help='The noise file to mix in')
parser.add_argument('--sample-rate', type = int, default=16000, help='Sample rate to save output as')
parser.add_argument('--noise-level', type=float, default=1.0,
                    help='The Signal to Noise ratio (higher means more noise)')
args = parser.parse_args()

data, sample_rate = dataset.read_wav(args.input_path)
data = torch.from_numpy(librosa.resample(data.numpy(), sample_rate, args.sample_rate))

noise, sample_rate = dataset.read_wav(args.noise_path)
noise = torch.from_numpy(librosa.resample(noise.numpy(), sample_rate, args.sample_rate))

noise = torch.cat([noise, torch.zeros_like(data[len(noise):])])

data = data * (1 - args.noise_level) + args.noise_level * noise

librosa.output.write_wav(args.output_path, data.numpy() / float(data.abs().max()), args.sample_rate)
