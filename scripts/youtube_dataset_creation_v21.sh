set -e 

CUDA_DEVICE=$1
THREAD=$2

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

YOUTUBE='https://www.youtube.com/channel/UCb9_Bhv37NXN1m8Bmrm9x9w'
CHECKPOINT=best_checkpoints/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs1536____rerun_base_model_from_scratch_epoch45_iter0230000.pt
SAMPLE_RATE=8000

SAMPLE=''
DATASET=youtube_v21
DATASET_ROOT=$DATASET
DATASET_AUDIO=$DATASET_ROOT/audio
DATASET_AUDIO_MASK=$DATASET_AUDIO/*/*
DATASET_TRANSCRIBE=$DATASET_ROOT/transcribe
DATASET_SUBSET=$DATASET_ROOT/subset
DATASET_CUT=$DATASET_ROOT/cut
DATASET_CUT_JSON=$DATASET_ROOT/cut/cut.json
DATASET_TRAIN_JSON=$DATASET_ROOT/cut/cut_train.json


TRANSCRIPT_PREPROC='--split-by-parts 4 --skip-files-longer-than-hours 4 --skip-transcript-after-seconds 3600'
TRANSCRIBE='--mono --join-transcript --batch-time-padding-multiple 1 --align --skip-processed --max-segment-duration 4.0 --transcribe-first-n-sec 3600 --text-pipelines dataset'
SUBSET='--gap 0.05- --cer 0.0-0.4 --duration 0.5-8.0'
CUT="--dilate 0.025 --sample-rate $SAMPLE_RATE --mono --strip-prefix data/ --add-sub-paths --strip"
TRAIN_TEST_SPLIT='--microval-duration-in-hours 10 --val-duration-in-hours 0 --test-duration-in-hours 0'

#mkdir -p $DATASET_AUDIO
#bash datasets/youtube.sh LIST "$YOUTUBE" > $DATASET_ROOT/youtube.txt
#bash datasets/youtube.sh LIST 'https://www.youtube.com/channel/UCb9_Bhv37NXN1m8Bmrm9x9w'
#head -n $SAMPLE $DATASET_ROOT/youtube.txt > $DATASET_AUDIO/audio.txt # list of links like http://youtu.be/e8bkFQD3ZhA
#bash datasets/youtube.sh RETR $DATASET_AUDIO/audio.txt $DATASET_AUDIO

#python3 datasets/youtube.py -i "$DATASET_AUDIO_MASK" -o $DATASET_AUDIO.json $TRANSCRIPT_PREPROC

#python3 transcribe.py --checkpoint $CHECKPOINT -i $DATASET_AUDIO.json -o $DATASET_TRANSCRIBE $TRANSCRIBE --output-json
#python3 transcribe.py --checkpoint $CHECKPOINT -i $DATASET_AUDIO$THREAD.json -o $DATASET_TRANSCRIBE $TRANSCRIBE --output-json

#python3 tools.py subset -i $DATASET_TRANSCRIBE -o $DATASET_SUBSET.json $SUBSET

#python3 tools.py cut -i $DATASET_SUBSET.json -o $DATASET_CUT $CUT

## if there are any empty ref if cut.json (empty refs can cause loss issue in train time) then remove it with cleanup_transcripts_with_empty_ref tool
# python tools.py cleanup_transcripts_with_empty_ref --input-path youtube_v21/cut/cut.json

python3 tools.py split -i $DATASET_CUT_JSON -o $DATASET_CUT $TRAIN_TEST_SPLIT

# in case you need 100/1000 hours dataset the script creates val and test
#python tools.py filter_dataset --input-path youtube_v21/cut/cut_train.json --output-path youtube_v21/cut/cut_train_cer_25_1000h.json --duration-in-hours 1000 --cer 25

#python3 vis.py audiosample -i $DATASET_CUT/$(basename $DATASET_CUT).json -o $DATASET_CUT.json.html
