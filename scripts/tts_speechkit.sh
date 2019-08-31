export INPUTFILE=$1
export OUTPUTDIR=$2
export NJOBS=${3:-20}
export SAMPLERATE=${4:-16000}
export FORMAT=${5:-lpcm}
export EXT=${6:-wav}


if [ -z $SPEECHKITAPIKEY ]; then
	SPEECHKITAPIKEY=$(cat speechkitapikey.txt)
fi
export SPEECHKITAPIKEY

random_voice() {
	P=(alyss jane oksana omazh zahar ermil)
	echo ${P[RANDOM%${#P[@]}]}
}
random_emotion() {
	P=(good evil neutral)
	echo ${P[RANDOM%${#P[@]}]}
}
random_speed() {
	P=(0.8 1.0 1.2)
	echo ${P[RANDOM%${#P[@]}]}
}

export -f random_voice
export -f random_emotion
export -f random_speed

# https://cloud.yandex.ru/docs/speechkit/tts/request
mkdir -p "$OUTPUTDIR"
cat "$INPUTFILE" | parallel --progress -j$NJOBS 'curl -s -X POST -H "Authorization: Api-Key $SPEECHKITAPIKEY" -d "lang=ru-RU&sampleRateHertz=$SAMPLERATE&format=$FORMAT&voice=$(random_voice)&emotion=$(random_emotion)&speed=$(random_speed)" "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize" --data-urlencode text={} | sox -r $SAMPLERATE -b 16 -e signed-integer -t raw - -c 1 -r $SAMPLERATE -t wav - > "$OUTPUTDIR/$(echo -n {} | md5sum | cut -d" " -f1).$EXT"'

while read line; do
	AUDIOPATH="$OUTPUTDIR/"$(echo -n $line | md5sum | cut -d" " -f1).$EXT
	DURATION=$(soxi -D "$AUDIOPATH")
	TRANSCRIPT=$(echo -n "$line" | tr -d "\n" | tr -d "\r")
	>&2 echo Computed duration of $AUDIOPATH
	echo "$AUDIOPATH,$TRANSCRIPT,$DURATION"
done < "$INPUTFILE" > "$OUTPUTDIR.csv"
