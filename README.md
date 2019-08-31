# convasr - Work In Progress
Baseline convolutional ASR system in PyTorch

# Dependencies
aria2 (for downloading **ru_open_stt** via torrent), PyTorch, NumPy, SciPy (for wav loading), librosa (for audio resampling), NVidia Apex (for fp16 training), tensorboard==1.14.0, future (PyTorch nightlies don't install future), spotty (for AWS Spot Instances training)
```shell
# installing apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex
```
Dependencies are also listed in `scripts/Dockerfile`. The **ru_open_stt** dataset download script is also in `scripts/download_ru_open_stt.sh`.

# Data file format
CSV (comma-separated) with 3 columns without header:
1. Full path to the audio wav file (mono, 16 Khz)
2. Transcript
3. Duration in seconds

# Usage
```shell
# transcribe all .wav in a directory and compute CER with .wav.txt, save transcription in .wav.transcript.txt
python3 transcribe.py \
  --checkpoint data/experiments/Wav2LetterRu_NovoGrad_lr1e-2_wd1e-3_bs80_augPSSPAMRNB0.5/checkpoint_epoch02_iter0074481.pt \
  --data-path data_dir
```

# Training on AWS Spot Instances with spotty
```shell
# download the ru_open_stt dataset on an AWS EBS volume from a t2.large On-Demand instance
spotty start -c scripts/spotty_preprocess.yaml
spotty run -c scripts/spotty_preprocess.yaml preprocess

# start a GPU instance
python scripts/spotty.py spotty start

# edit scripts/train.sh and launch training on a GPU instance
python scripts/spotty.py train scripts/train.sh

# check CER
python scripts/spotty.py cer EXPERIMENT_ID --val-dataset-name clean_val.csv

# download a checkpoint
python scripts/spotty.py download_checkpoint CHECKPOINT_PATH

# check spot instance prices
spotty aws spot-prices -i p3.8xlarge -r us-east-1
```

# Augment a dataset with SOX (offline)
The passed command must read from stdin and write to stdout.

```shell
# encode to GSM and back
bash scripts/augment.sh data/clean_val.csv data/clean_val_gsm "sox -V0 -t wav - -r 8k -c 1 -t gsm - | sox -V0 -r 8k -t gsm - -t wav -b 16 -e signed -r 8k -c 1 -"

# encode to AMR (NB: narrow-band, 8kHz) and back
bash scripts/augment.sh data/clean_val.csv data/clean_val_amrnb "sox -V0 -t wav - -r 8k -c 1 -t amr-nb - | sox -V0 -r 8k -t amr-nb - -t wav -b 16 -e signed -r 8k -c 1 -"

# denoise with RNNnoise
LD_LIBRARY_PATH=rnnoise/.libs bash scripts/augment.sh ../sample_ok/sample_ok.convasr.csv data/sample_ok.convasr.rnnoise "sox -t wav - -r 48k --bits 16 --encoding signed-integer --endian little -t raw - | ./rnnoise/examples/rnnoise_demo /dev/stdin /dev/stdout | sox -t raw -r 48k --encoding signed-integer --endian little --bits 16 - -t wav -b 16 -e signed -r 16k -c 1 -"

# denoise with SOX
sox data/noise/1560751355.653399.wav_1.wav -n noiseprof data/noise.prof
bash scripts/augment.sh ../sample_ok/sample_ok.convasr.csv data/sample_ok.convasr.noisered "sox -V0 -t wav - -t wav - noisered data/noise.prof 0.5"

# transcode OGG to WAV
bash scripts/augment.sh data/speechkit.csv data/speechkit_wav "opusdec - --quiet --force-wav -"

# print duration of a data file in hours
bash script/duration.sh data/mixed_train.csv
```

# Docker commands
```shell
# build scripts/Dockerfile
sudo nvidia-docker build -t convasr scripts

# run docker
sudo nvidia-docker run -v $PWD/deepspeech.pytorch:/deepspeech.pytorch -it --ipc=host convasr 
```

# KenLM
Dependencies: `sudo apt-get install build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev`
```shell
# build kenlm
wget https://github.com/kpu/kenlm/archive/master.tar.gz -O kenlm.tar.gz
tar -xf kenlm.tar.gz
cd master
mkdir build
cd build
cmake ..
make -j 4

# estimate model in the text ARPA format
bin/lmplz -o 4 <text.csv >lm.arpa

# binarize the estimated ARPA model
bin/build_binary /dev/stdin lm.bin <lm.arpa
```

# Beam search decoder
Dependencies: same as KenLM, `pip install wget`
```shell
pip install git+https://github.com/parlance/ctcdecode
```

# TTS generation
```shell
# select top messages approx 700k cyrillic chars (a cyrillic char takes 2 bytes in UTF-8, hence x2 factor), dropping the last line
head -c 1500000 tts_dataset.txt | head -n -1 > tts_dataset_100h.txt

# using yandex speehkit tts with OGG
# make sure your apikey has necessary roles, e.g. admin
# export the API key or create a speechkitapikey.txt file

# run generation in ogg format with 10 workers
export SPEECHKITAPIKEY=yourapikey
bash scripts/tts_speechkit.sh tts_dataset_100h.txt data/speechkit
```
