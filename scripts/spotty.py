import os
import argparse
import subprocess

spotty_yaml = os.path.join(os.path.dirname(__file__), 'spotty.yaml')

def start():
    subprocess.call(['spotty', 'start', '-c', spotty_yaml])

def stop():
    subprocess.call(['spotty', 'stop', '-c', spotty_yaml])

def train(script):
    lines = [l.strip() for l in open(script) if l.strip() and not l.startswith('#')]
    first = [i for i, l in enumerate(lines) if 'train.py' in l][0]
    last = [i for i in range(first + 1, len(lines)) if lines[i][-1] == '\\' or lines[i - 1][-1] == '\\'][-1]
    ARGS = ''.join(l.rstrip('\\') for l in lines[first + 1:last + 1]); print(ARGS)
    subprocess.call(['spotty', 'run', '-c', spotty_yaml, 'train', '-p', 'ARGS=' + ARGS])

def download_checkpoint(id):
    subprocess.call(['spotty', 'download', '-c', spotty_yaml, '-f', os.pathjoin('experiments', id)])

def cer(id, val_dataset_name):
    subprocess.call(['spotty', 'run', '-c', spotty_yaml, 'cer', '-p', 'ID=' + id, 'VALDATASETNAME=' + val_dataset_name])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    cmd = subparsers.add_parser('start')
    cmd.add_argument('start')
    cmd.set_defaults(func = start)

    cmd = subparsers.add_parser('stop')
    cmd.add_argument('stop')
    cmd.set_defaults(func = stop)

    cmd = subparsers.add_parser('train')
    cmd.add_argument('script')
    cmd.set_defaults(func = train)

    cmd = subparsers.add_parser('download_checkpoint')
    cmd.add_argument('--id', required = True)
    cmd.set_defaults(func = download_checkpoint)
    
    cmd = subparsers.add_parser('cer')
    cmd.add_argument('--id', required = True)
    cmd.add_argument('--val-dataset-name', default = 'clean_val.csv')
    cmd.set_defaults(func = cer)

    args = vars(parser.parse_args())
    func = args.pop('func')
    func(**args)
