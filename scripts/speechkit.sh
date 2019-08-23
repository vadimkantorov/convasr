export INPUTFILE=$1
export OUTPUTDIR=$2
export NJOBS=${3:-20}
export SAMPLERATE=${4:-16000}
export FORMAT=${5:-oggopus}
export EXT={6:-ogg}

if [ -z $SPEECHKITAPIKEY ]; then
	SPEECHKITAPIKEY=$(cat speechkitapikey.txt)
fi

random_voice() {
	P=(alyss jane oksana omazh zahar ermil)
	echo ${P[RANDOM%${#P[@]}]}
}
random_emotion() {
	P=(good evil neutral)
	echo ${P[RANDOM%${#P[@]}]}
}
random_speed() {
	P=(0.7 0.8 1.0)
	echo ${P[RANDOM%${#P[@]}]}
}

export -f random_voice
export -f random_emotion
export -f random_speed

# https://cloud.yandex.ru/docs/speechkit/tts/request
mkdir -p "$OUTPUTDIR"
cat "$INPUTFILE" | parallel -j$NJOBS 'curl -s -X POST -H "Authorization: Api-Key $SPEECHKITAPIKEY"  -d "lang=ru-RU&sampleRateHergz=$SAMPLERATE&format=$FORMAT&voice=$(random_voice)&emotion=$(random_emotion)&speed=$(random_speed)" https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize --data-urlencode text={} > "$OUTPUTDIR/$(echo -n {} | md5sum | cut -d" " -f1).$EXT"'
