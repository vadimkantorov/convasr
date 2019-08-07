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
