set -e 

export CUDA_VISIBLE_DEVICES=1

YOUTUBE='https://www.youtube.com/channel/UCb9_Bhv37NXN1m8Bmrm9x9w'
CHECKPOINT=best_checkpoints/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs1024____long-train_bs1024_step6_430k_checkpoint_epoch247_iter0446576.pt
SAMPLE_RATE=8000

SAMPLE=''
DATASET=youtube$SAMPLE
DATASET_ROOT=data/$DATASET
DATASET_AUDIO=$DATASET_ROOT/audio
DATASET_TRANSCRIBE=$DATASET_ROOT/transcribe
DATASET_SUBSET=$DATASET_ROOT/subset
DATASET_CUT=$DATASET_ROOT/cut

TRANSCRIBE='--mono --align --max-segment-duration 4.0 --html'
SUBSET='--align-boundary-words --num-speakers 1 --gap 0.1- --cer 0.1-0.4 --duration 2.0-4.0'
CUT="--dilate 0.02 --sample-rate $SAMPLE_RATE --mono"

#mkdir -p $DATASET_AUDIO
#bash datasets/youtube.sh LIST "$YOUTUBE" > $DATASET_ROOT/youtube.txt
#bash datasets/youtube.sh LIST 'https://www.youtube.com/channel/UCb9_Bhv37NXN1m8Bmrm9x9w'
#head -n $SAMPLE $DATASET_ROOT/youtube.txt > $DATASET_AUDIO/audio.txt # list of links like http://youtu.be/e8bkFQD3ZhA
#bash datasets/youtube.sh RETR $DATASET_AUDIO/audio.txt $DATASET_AUDIO

#python3 datasets/youtube.py -i $DATASET_AUDIO -o $DATASET_AUDIO

python3 transcribe.py --checkpoint $CHECKPOINT -i $DATASET_AUDIO -o $DATASET_TRANSCRIBE $TRANSCRIBE

#python3 tools.py subset -i $DATASET_TRANSCRIBE -o $DATASET_SUBSET.json $SUBSET
#python3 tools.py cut -i $DATASET_SUBSET.json -o $DATASET_CUT $CUT
#python3 vis.py audiosample -i $DATASET_CUT/$(basename $DATASET_CUT).json -o $DATASET_CUT.json.html
