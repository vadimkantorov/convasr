INPUTFILE=$1
OUTPUTDIR=$2

if [ -z $SPEECHKITAPIKEY ]; then
	SPEECHKITAPIKEY=$(cat speechkitapikey.txt)
fi

SAMPLERATE=16000
FORMAT=oggopus
EXT=ogg

# https://cloud.yandex.ru/docs/speechkit/stt/request

NEEDCOMMA=
echo "{" > "$OUTPUTDIR.csv"
while read line; do
	AUDIOPATH=$(echo -n $line | cut -d "," -f 1)
	REFERENCE=$(echo -n $line | cut -d "," -f 2)
	DURATION=$(echo -n $line | cut -d "," -f 3)
	TRANSCRIPT=$(sox -V0 "$AUDIOPATH" -c 1 -r $SAMPLERATE -t $EXT - | curl -s -X POST -H "Authorization: Api-Key $SPEECHKITAPIKEY" --data-binary @- -d "lang=ru-RU&sampleRateHertz=$SAMPLERATE&format=$FORMAT&raw_results=true" "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize" | grep result | cut -d'"' -f4)
	if [ $NEEDCOMMA ]; then	echo ","; fi
	echo "{\"\filename\" : \"$AUDIOPATH\", \"reference\" : \"$REFERENCE\", \"transcript\" : \"$TRANSCRIPT\"}"
	NEEDCOMMA=1
done < "$INPUTFILE" >> "$OUTPUTDIR".csv
echo "}" >> "$OUTPUTDIR.csv"
