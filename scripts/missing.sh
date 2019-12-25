set -e

INPUT=${1:-data/speechcore/openstt_bad_model.json} 
LO=${2:-0.3}
HI=${3:-0.5}

OUTPUTMIN=$(python3 metrics.py subset $INPUT --arg cer --min $HI)
OUTPUTMAX=$(python3 metrics.py subset $INPUT --arg cer --max $LO)
OUTPUTMINMAX=$(python3 metrics.py subset $INPUT --arg cer --min $LO --max $HI)

echo $OUTPUTMIN && python3 vis.py errors $INPUT --audio-file-name $OUTPUTMIN --audio
echo $OUTPUTMAX && python3 vis.py errors $INPUT --audio-file-name $OUTPUTMAX --audio
echo $OUTPUTMINMAX && python3 vis.py errors $INPUT --audio-file-name $OUTPUTMINMAX --audio

