import argparse
import os
from pydub import AudioSegment
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dirs', nargs='+', help='paths to the input directories')
    parser.add_argument('--output-dir', help='path to the output directory')
    parser.add_argument('--filenames-file', help='path to the file with audio files names')
    args = parser.parse_args()

    input_dirs = args.input_dirs
    output_dir = args.output_dir
    #filenames_file = args.filenames_file

    #with open(filenames_file) as file:
    #    filenames = file.read().splitlines()

    #filenames = set(filenames)

    os.makedirs(output_dir, exist_ok=True)

    for input_dir in input_dirs:
        for filename in tqdm(os.listdir(input_dir)):

            #if filename not in filenames:
            #    continue

            sound = AudioSegment.from_wav(os.path.join(input_dir, filename))

            channels = sound.split_to_mono()

            channels[0].export(os.path.join(output_dir, '0_' + filename), format='wav')
            channels[1].export(os.path.join(output_dir, '1_' + filename), format='wav')


if __name__ == '__main__':
    main()
