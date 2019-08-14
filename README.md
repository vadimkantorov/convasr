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
bash scripts/augment.sh data/clean_val.csv data/clean_val_gsm "sox -V0 -t wav - -r 8k -c 1 -t gsm - | sox -V0 -t gsm - -t wav -b 16 -e signed -r 16k -c 1 -"

# encode to AMR (NB: narrow-band, 8kHz) and back
bash scripts/augment.sh data/clean_val.csv data/clean_val_amrnb "sox -V0 -t wav - -r 8k -c 1 -t amr-nb - | sox -V0 -t amr-nb - -t wav -b 16 -e signed -r 16k -c 1 -"
```

# Docker commands
```
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
bin/lmplz -o 2 <text.csv >lm.arpa
bin/build_binary /dev/stdin lm.bin <lm.arpa
```

# Beam search decoder
Dependencies: same as KenLM, `pip install wget`
```shell
pip install git+https://github.com/parlance/ctcdecode
```
