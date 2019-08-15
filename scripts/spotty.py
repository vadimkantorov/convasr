#! /usr/bin/python3

import os
import argparse
import subprocess


def spotty(spotty_yaml, arguments):
    subprocess.call(['spotty', arguments[0], '-c', spotty_yaml] + arguments[1:])

def train(spotty_yaml, script):
    lines = [l.strip() for l in open(script) if l.strip() and not l.startswith('#')]
    first = [i for i, l in enumerate(lines) if 'train.py' in l][0]
    last = [i for i in range(first + 1, len(lines)) if lines[i][-1] == '\\' or lines[i - 1][-1] == '\\'][-1]
    ARGS = ''.join(l.rstrip('\\') for l in lines[first + 1:last + 1]); print(ARGS)
    subprocess.call(['spotty', 'run', '-c', spotty_yaml, 'train', '-p', 'ARGS=' + ARGS])

def download_checkpoint(spotty_yaml, checkpoint_path):
    subprocess.call(['spotty', 'download', '-c', spotty_yaml, '-f', os.path.join('experiments', checkpoint_path)])

def cer(experiment_id, val_dataset_name, spotty_yaml):
    subprocess.call(['spotty', 'run', '-c', spotty_yaml, 'cer', '-p', 'ID=' + experiment_id, 'VALDATASETNAME=' + val_dataset_name])

def tensorboard(spotty_yaml, experiment_id):
    subprocess.call(['spotty', 'run', '-c', spotty_yaml, 'tensorboard', '-p', 'ID=' + experiment_id])

if __name__ == '__main__':
    spotty_yaml = os.path.join(os.path.dirname(__file__), 'spotty.yaml')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--spotty_yaml', default = spotty_yaml)
    subparsers = parser.add_subparsers()

    cmd = subparsers.add_parser('spotty')
    cmd.add_argument('arguments', nargs = argparse.REMAINDER)
    cmd.set_defaults(func = spotty)

    cmd = subparsers.add_parser('train')
    cmd.add_argument('script')
    cmd.set_defaults(func = train)

    cmd = subparsers.add_parser('download_checkpoint')
    cmd.add_argument('checkpoint_path')
    cmd.set_defaults(func = download_checkpoint)
    
    cmd = subparsers.add_parser('cer')
    cmd.add_argument('experiment_id')
    cmd.add_argument('-d', '--val-dataset-name', default = 'clean_val.csv')
    cmd.set_defaults(func = cer)

    cmd = subparsers.add_parser('tensorboard')
    cmd.add_argument('experiment_id')
    cmd.set_defaults(func = tensorboard)

    args = vars(parser.parse_args())
    func = args.pop('func')
    func(**args)
