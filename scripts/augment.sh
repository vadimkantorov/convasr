INPUTFILE=$1
OUTPUTDIR=$2
shift 2
CMD=$@

mkdir -p "$OUTPUTDIR"
while read line; do
    INPUTWAV=$(echo $line | cut -d ',' -f1)
    REST=$(echo $line | cut -d ',' -f2-3)
    OUTPUTWAV=$OUTPUTDIR/$(basename "$INPUTWAV")
    bash -c "$CMD" >$OUTPUTWAV <$INPUTWAV
	>&2 echo Converted $INPUTWAV to $OUTPUTWAV
    echo "$OUTPUTWAV,$REST"
done < "$INPUTFILE" > "$OUTPUTDIR.csv"
