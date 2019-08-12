INPUTFILE=$1
OUTPUTDIR=$2
shift 2
CMD=$@

mkdir -p "$OUTPUTDIR"
while read line; do
    INPUTWAV=$(echo $line | cut -d ',' -f1)
    REST=$(echo $line | cut -d ',' -f2-3)
    OUTPUTWAV=$OUTPUTDIR/$(basename "$INPUTWAV")
    CMDFIXED=${CMD/input.wav/$INPUTWAV}
    CMDFIXED=${CMDFIXED/output.wav/$OUTPUTWAV}
#    >&2 echo $CMDFIXED
    bash -c "$CMDFIXED" >&2
    echo "$OUTPUTWAV,$REST"
done < "$INPUTFILE" > "$OUTPUTDIR.csv"
