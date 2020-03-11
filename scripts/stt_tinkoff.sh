INPUTFILE=$1
OUTPUTDIR=$2
OUTPUTFILE="$(dirname $OUTPUTDIR)/transcripts_tinkoff_$(basename $OUTPUTDIR).json"
TINKOFFSPEECHAPIEXAMPLESGIT=tinkoff-speech-api-examples

if [ -z $TINKOFFAPIKEY ]; then
	TINKOFFAPIKEY=$(cat tinkoffapikey.txt)
fi
if [ -z $TINKOFFSECRETKEY ]; then
	TINKOFFSECRETKEY=$(cat tinkoffsecretkey.txt)
fi


SAMPLERATE=16000
ENCODING=A-LAW

# https://voicekit.tinkoff.ru/docs/recognition
# https://voicekit.tinkoff.ru/docs/usingstt
# https://github.com/TinkoffCreditSystems/tinkoff-speech-api-examples/blob/master/sh/recognize.sh

if [ ! -f $TINKOFFSPEECHAPIEXAMPLESGIT ]; 
	git clone https://github.com/TinkoffCreditSystems/tinkoff-speech-api-examples
fi
export PYTHONPATH=$TINKOFFSPEECHAPIEXAMPLESGIT/python:$PYTHONPATH
while read line; do
	AUDIOPATH=$(echo -n $line | cut -d "," -f 1)
	REF=$(echo -n $line | cut -d "," -f 2)
	HYP=$(sox -V0 "$AUDIOPATH" -c 1 -r $SAMPLERATE -t raw -e $ENCODING - | python3 -m recognize --host stt.tinkoff.ru --port 443 --api_key $TINKOFFAPIKEY --secret_key $TINKOFFSECRETKEY --rate $SAMPLERATE --num_channels 1 --encoding ${ENCODING/-/} --audio_file /dev/stdin | grep Transcription | sed 's/Transcription //')
	echo "{\"audio_path\" : \"$AUDIOPATH\", \"ref\" : \"$REF\", \"hyp\" : \"$HYP\"}"
done < "$INPUTFILE" | (echo '['; paste -sd ',' -; echo ']') > "$OUTPUTFILE"
