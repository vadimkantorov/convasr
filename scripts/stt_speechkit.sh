INPUTFILE=$1
OUTPUTDIR=$2

if [ -z $SPEECHKITAPIKEY ]; then
	SPEECHKITAPIKEY=$(cat speechkitapikey.txt)
fi

SAMPLERATE=16000
FORMAT=oggopus
EXT=ogg

# https://cloud.yandex.ru/docs/speechkit/stt/request

while read line; do
	AUDIOPATH=$(echo -n $line | cut -d "," -f 1)
	DURATION=$(echo -n $line | cut -d "," -f 3)
	TRANSCRIPT=$(sox -V0 "$AUDIOPATH" -c 1 -r $SAMPLERATE -t $EXT - | curl -s -X POST -H "Authorization: Api-Key $SPEECHKITAPIKEY" --data-binary @- -d "lang=ru-RU&sampleRateHertz=$SAMPLERATE&format=$FORMAT&raw_results=true" "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize" | grep result | cut -d'"' -f4)
	echo "$AUDIOPATH,$TRANSCRIPT,$DURATION"
done < "$INPUTFILE" > "$OUTPUTDIR".csv
