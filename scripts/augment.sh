INPUTFILE=$1
OUTPUTDIR=$2
shift 2
CMD=$@

mkdir -p "$OUTPUTDIR"
while read line; do
    AUDIOPATH=$(echo $line | cut -d ',' -f1)
    REST=$(echo $line | cut -d ',' -f2-3)
    OUTPUTWAV=$OUTPUTDIR/$(basename "$AUDIOPATH")
    bash -c "$CMD" >$OUTPUTWAV <$AUDIOPATH
	>&2 echo Converted $AUDIOPATH to $OUTPUTWAV
    echo "$OUTPUTWAV,$REST"
done < "$INPUTFILE" > "$OUTPUTDIR.csv"
