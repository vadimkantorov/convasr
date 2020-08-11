#!/bin/bash

DIR=$1

for f in "$DIR"/*/*/*; do
  if [ -d "$f" ]; then
    for file in "$f"/*.wav; do
      (
      sourcefile=$file
      filename="${file%.*}"
      destfile="$filename.dest.wav"

      echo $sourcefile;
      ffmpeg -loglevel error -y -i $sourcefile -vn -ar 8000 -ac 1 -filter:a "highpass=f=100, lowpass=f=2000" $destfile
      ffmpeg -loglevel error -y -i $destfile -v quiet -acodec pcm_s16le -ar 8000 -ac 1 $sourcefile
      rm $destfile
      ) &
    done
    wait
  fi
done


