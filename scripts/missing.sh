set -e

INPUT=$1 
LO=${2:-0.3}
HI=${3:-0.5}

OUTPUTMIN=$(python3 vis.py subset $INPUT --arg cer --min $HI)
OUTPUTMAX=$(python3 vis.py subset $INPUT --arg cer --max $HI)
OUTPUTMINMAX=$(python3 vis.py subset $INPUT --arg cer --min $LO --max $HI)

echo $OUTPUTMIN && python3 vis.py errors $INPUT --audio-file-name $OUTPUTMIN --audio
echo $OUTPUTMAX && python3 vis.py errors $INPUT --audio-file-name $OUTPUTMAX --audio
echo $OUTPUTMINMAX && python3 vis.py errors $INPUT --audio-file-name $OUTPUTMINMAX --audio

