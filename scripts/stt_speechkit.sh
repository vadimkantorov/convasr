INPUTFILE=$1
OUTPUTDIR=$2
OUTPUTFILE="$(dirname $OUTPUTDIR)/transcripts_speechkit_$(basename $OUTPUTDIR).json"

if [ -z $SPEECHKITAPIKEY ]; then
	SPEECHKITAPIKEY=$(cat speechkitapikey.txt)
fi

SAMPLERATE=16000
FORMAT=oggopus
EXT=ogg

# https://cloud.yandex.ru/docs/speechkit/stt/request
while read line; do
	AUDIOPATH=$(echo -n $line | cut -d "," -f 1)
	REFERENCE=$(echo -n $line | cut -d "," -f 2)
	DURATION=$(echo -n $line | cut -d "," -f 3)
	TRANSCRIPT=$(sox -V0 "$AUDIOPATH" -c 1 -r $SAMPLERATE -t $EXT - | curl -s -X POST -H "Authorization: Api-Key $SPEECHKITAPIKEY" --data-binary @- -d "lang=ru-RU&sampleRateHertz=$SAMPLERATE&format=$FORMAT&raw_results=true" "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize" | grep result | cut -d'"' -f4)
	echo "{\"filename\" : \"$AUDIOPATH\", \"reference\" : \"$REFERENCE\", \"transcript\" : \"$TRANSCRIPT\"}"
done < "$INPUTFILE" | (echo '['; paste -sd ',' -; echo ']') > "$OUTPUTFILE"
