INPUTFILE=$1
OUTPUTDIR=$2
OUTPUTFILE="$(dirname $OUTPUTDIR)/transcripts_tinkoff_$(basename $OUTPUTDIR).json"
TINKOFFSPEECHAPIEXAMPLESGIT=./tinkoff-speech-api-examples

if [ -z $TINKOFFAPIKEY ]; then
	TINKOFFAPIKEY=$(cat tinkoffapikey.txt)
fi
if [ -z $TINKOFFSECRETKEY ]; then
	TINKOFFSECRETKEY=$(cat tinkoffsecretkey.txt)
fi

SAMPLERATE=16000
FORMAT=raw
EXT=raw
ENCODING=A-LAW

# https://github.com/TinkoffCreditSystems/tinkoff-speech-api-examples/blob/master/sh/recognize.sh

export PYTHONPATH=$TINKOFFSPEECHAPIEXAMPLESGIT/python:$PYTHONPATH
while read line; do
	AUDIOPATH=$(echo -n $line | cut -d "," -f 1)
	REFERENCE=$(echo -n $line | cut -d "," -f 2)
	DURATION=$(echo -n $line | cut -d "," -f 3)
	sox "$AUDIOPATH" -c 1 -r $SAMPLERATE -t $FORMAT -e $ENCODING tmp.$EXT
	
	TRANSCRIPT=$(python3 -m recognize --host stt.tinkoff.ru --port 443 --api_key $TINKOFFAPIKEY --secret_key $TINKOFFSECRETKEY --rate $SAMPLERATE --num_channels 1 --encoding ${ENCODING/-/} --audio_file tmp.$EXT | grep Transcription | sed 's/Transcription //')
	echo "{\"filename\" : \"$AUDIOPATH\", \"reference\" : \"$REFERENCE\", \"transcript\" : \"$TRANSCRIPT\"}"
	break
done < "$INPUTFILE" #| (echo '['; paste -sd ',' -; echo ']') > "$OUTPUTFILE"
