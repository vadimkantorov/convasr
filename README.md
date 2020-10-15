# convasr - Work In Progress
Baseline convolutional ASR system in PyTorch

# License
MIT

# Dependencies
aria2 (for downloading **ru_open_stt** via torrent), PyTorch, NumPy, SciPy (for wav loading), librosa (for audio resampling), NVidia Apex (for fp16 training), tensorboard==1.14.0, future (PyTorch nightlies don't install future), spotty (for AWS Spot Instances training)
```shell
# installing apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex
```
Dependencies are also listed in `scripts/Dockerfile`. The **ru_open_stt** dataset download script is also in `scripts/download_ru_open_stt.sh`.

# Usage
```shell
python3 transcribe.py \
  --checkpoint data/experiments/Wav2LetterRu_NovoGrad_lr1e-2_wd1e-3_bs80_augPSSPAMRNB0.5/checkpoint_epoch02_iter0074481.pt \
  -i data_dir
```

# BPE pretrained models for Russian
```shell
python tools.py bpetrain -i data/tts_dataset/tts_dataset.txt -o data/tts_dataset_bpe_1000 --vocab-size 1000
python tools.py bpetrain -i data/tts_dataset/tts_dataset.txt -o data/tts_dataset_bpe_5000 --vocab-size 5000
python tools.py bpetrain -i data/tts_dataset/tts_dataset.txt -o data/tts_dataset_bpe_10000 --vocab-size 10000

# from https://nlp.h-its.org/bpemb/ru/
wget https://nlp.h-its.org/bpemb/ru/ru.wiki.bpe.vs5000.vocab -P data
wget https://nlp.h-its.org/bpemb/ru/ru.wiki.bpe.vs5000.model -P data
```

# Debugging ONNX problems
```shell
# export to ONNX and disable exporting weights for minimal file size
python3 train.py --onnx data/model.onnx --onnx-export-params=

# upload data/model.onnx to https://lutzroeder.github.io/netron/
```

# Format code
```shell
bash scripts/fmtall.sh
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
# transcode to MP3
python3 tools.py transcode -o calls_micro/calls_micro.json -o data/calls_micro_mp3 --ext .mp3 "sox -V0 -t wav - -r 8k -t mp3 -"

# encode to GSM and back
"sox -V0 -t wav - -r 8k -c 1 -t gsm - | sox -V0 -r 8k -t gsm - -t wav -b 16 -e signed -r 8k -c 1 -"

# encode to AMR (NB: narrow-band, 8kHz) and back
"sox -V0 -t wav - -r 8k -c 1 -t amr-nb - | sox -V0 -r 8k -t amr-nb - -t wav -b 16 -e signed -r 8k -c 1 -"

# denoise with RNNnoise
export LD_LIBRARY_PATH=rnnoise/.libs
"sox -t wav - -r 48k --bits 16 --encoding signed-integer --endian little -t raw - | ./rnnoise/examples/rnnoise_demo /dev/stdin /dev/stdout | sox -t raw -r 48k --encoding signed-integer --endian little --bits 16 - -t wav -b 16 -e signed -r 8k -c 1 -"

# denoise with SOX
sox data/noise/1560751355.653399.wav_1.wav -n noiseprof data/noise.prof
"sox -V0 -t wav - -t wav - noisered data/noise.prof 0.5"

# transcode OGG to WAV
"opusdec - --quiet --force-wav -"

# convert s16le to f32le
"sox -V0 -t wav - -r 16k -b 32 -e float -t wav -c 1 -"

# denoise with https://github.com/francoisgermain/SpeechDenoisingWithDeepFeatureLosses
python senet_infer.py -d ../data/sample_ok.convasr.1.sox -m models

# convert f32le to s16le
"sox -V0 -t wav - -r 16k -b 16 -e signed -t wav -c 1 -"

# print total duration of audio files in a directory
find -type f -name '*.wav' | xargs soxi -D | awk '{sum += $1} END {print sum / 3600; print "hours"}'
```

# Docker commands
```shell
# build scripts/Dockerfile
sudo docker build --build-arg CUDAVERSION=101 --build-arg CUDAVERSIONPOINT=10.1 -t convasr scripts
sudo docker build --build-arg CUDAVERSION=102 --build-arg CUDAVERSIONPOINT=10.2 -t convasr scripts

# run docker
sudo docker run -p 7006:6006 --runtime=nvidia --privileged --cap-add=SYS_PTRACE -v ~/.ssh:/root/.ssh -v ~/stt_results:/root/stt_results -v ~/.gitconfig:/root/.gitconfig -v ~/.vimrc:/root/.vimrc -v ~/convasr:/root/convasr -v /home/data/ru_open_stt_wav:/root/convasr/ru_open_stt -v /home/data/kontur_calls_micro:/root/convasr/kontur_calls_micro -v /home/data/valset17122019:/root/convasr/valset17122019 -v /home/data/valset11102019:/root/convasr/valset11102019 -v /home/data/domain_set:/root/convasr/domain_set -v /home/data/speechcore:/root/convasr/data/speechcore -v/home/html:/root/convasr/data/html -v /home/data/youtube:/root/convasr/youtube -it --ipc=host convasr

sudo docker image rm -f convasr
```

# ssh port forwarding
```shell
ssh -L 7006:$HOST:7006 $HOST -N # -f for background
```

# Tensorboard
```shell
ssh -L 6007:YOURHOST:6006 YOURHOST -N &
```

# Html server docker commands
```shell
# launch nginx docker
docker run -d -p 127.0.0.1:8080:80 -v /home/data/html:/usr/share/nginx/html:ro -it --rm nginx

# ssh port forwarding for port 8080
ssh -L 8081:YOURHOST:8080 YOURHOST -N &
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

# extract training transcripts
cut -d',' -f 2 data/mixed_train.csv > data/mixed_train.txt
```

# Beam search decoder
Dependencies: same as KenLM, `pip install wget`
```shell
pip install git+https://github.com/parlance/ctcdecode
```

# TTS generation
```shell
# select top messages approx 700k cyrillic chars (a cyrillic char takes 2 bytes in UTF-8, hence x2 factor), dropping the last line
head -c 1500000 tts_dataset.txt | head -n -1 > tts_dataset_15h.txt

# split giant dataset in chunks of 15h
mkdir -p data/tts_dataset_splits && split --lines 12000 --numeric-suffixes --suffix-length 4 tts_dataset.txt data/tts_dataset_splits/tts_dataset.txt_

# using yandex speehkit tts with OGG
# make sure your apikey has necessary roles, e.g. admin
# export the API key or create a speechkitapikey.txt file

# run generation in ogg format with 10 workers
export SPEECHKITAPIKEY=yourapikey
bash scripts/tts_speechkit.sh tts_dataset_100h.txt data/speechkit
```

# Download unigram frequency list
```shell
# for russian
wget http://opencorpora.org/files/export/ngrams/unigrams.cyr.lc.bz2 -P data
bzip2 -d data/unigrams.cyr.lc.bz2

wget https://github.com/Koziev/NLP_Datasets/raw/master/WordformFrequencies/Data/term2freq.7z -P data
7z x data/term2freq.7z -odata
```

# Serving mock API of Google Cloud Speech API (only for testing)
```shell
# serve
python3 serve_google_api.py --endpoint localhost:50051 --checkpoint ...

# test
python3 scripts/stt_google.py --endpoint localhost:50051 --lang ru --api-key-credentials= -i calls_micro/calls_micro.json
```

# Frontend performance metrics
```shell
bash scripts/read_audio_performance.sh
```
| file       | reads count |  backend | process_time us| perf_counter us| 
|-----------:|---:|---------:|--------------:|-------------:|
|data/tests/test_5s.wav|  100|       sox|   431341|    11886|
|data/tests/test_1m.wav|  100|       sox|   455051|    12593|
|data/tests/test_1h.wav|  100|       sox|  8939791|   458676|
|data/tests/test_5s.wav|  100|    ffmpeg|  5147064|   140222|
|data/tests/test_1m.wav|  100|    ffmpeg|  3941069|   306300|
|data/tests/test_1h.wav|  100|    ffmpeg| 10509560|   628091|
|data/tests/test_5s.wav|  100| soundfile|    42835|     1680|
|data/tests/test_1m.wav|  100| soundfile|    36295|     1006|
|data/tests/test_1h.wav|  100| soundfile|  4311895|   215836|
|data/tests/test_5s.wav|  100|     scipy|    30163|     1583|
|data/tests/test_1m.wav|  100|     scipy|    35958|     1092|
|data/tests/test_1h.wav|  100|     scipy|  3579850|   215113|

# Configuring Jigasi Meet transcription for Jitsi
[Docs](https://nikvaessen.github.io/jekyll/update/2017/08/24/gsoc2017_work_product_submission.html)

[Configuration](https://github.com/jitsi/jigasi#using-jigasi-to-transcribe-a-jitsi-meet-conference)

# Text processing dataflow

All text processing are packed into pipelines. 

`pipeline` is an object that implements methods:
1. `preprocess(text: str) -> str` method for text preprocessing
2. `postprocess(text: str) -> str` method for text postprocessing (assumed that input was preprocessed)
3. `encode(texts: List[str]) -> List[List[int]]` method to convert texts into tokens
4. `decode(texts: List[List[int]]) -> List[str]` method to convert tokens into texts


## Train/Metrics

dataset.py:
1. Read `REF` and store it into `meta` dict of dataset example.
2. Make `TARGET` for model training. 

`REF -> pipeline.preprocess -> pipeline.encode -> TARGET`

train.py:
3. Get `log_probs` from model.
4. Get `transcrips` from generator.
5. Built `HYP` by concatenation of transcript segments.
6. Apply `pipeline.preprocess` to REF. It is necessary for validation, because future postprocessing in validation assumed that text was preprocessed.

`REF -> pipeline.preprocess -> REF`

metrics.py:
7. Apply `pipeline.postprocess` to both HYP and REF. 

`HYP -> pipeline.postprocess -> HYP`
`REF -> pipeline.postprocess -> REF`

8. If val_config contains some additional postprocessor apply it to both HYP and REF.

`HYP -> postprocessor -> HYP`
`REF -> postprocessor -> REF`

9. Compute metrics in according with config. 

## Transcribe

dataset.py:
1. Read `REF` and store it into `meta` dict of dataset example.
transcribe.py
3. Get `log_probs` from model.
4. Get `transcrips` from generator.
5. Built `HYP` by concatenation of transcript segments.
6. Apply `pipeline.preprocess` to REF.

`REF -> pipeline.preprocess -> REF`

7. Apply `pipeline.postprocess` to both HYP and REF. 

`HYP -> pipeline.postprocess -> HYP`
`REF -> pipeline.postprocess -> REF`

8. Write `HYP` and `REF` into file.