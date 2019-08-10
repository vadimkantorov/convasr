INPUTFILE=$1
OUTPUTDIR=$2
shift 2

mkdir -p "$OUTPUTDIR"
while read line; do
    WAVPATH=$(echo $line | cut -d ',' -f1)
    REST=$(echo $line | cut -d ',' -f2-3)
    NEWPATH=$OUTPUTDIR/$(basename "$WAVPATH")
    cp "$WAVPATH" input.wav
    $@
    mv output.wav "$OUTPUTDIR" && rm input.wav
    echo "$NEWPATH,$REST"
done < "$INPUTFILE" > "$OUTPUTDIR.csv"
