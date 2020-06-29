#!/bin/bash

DIR=$1

for f in "$DIR"/*/*/*; do
  if [ -d "$f" ]; then
    for file in "$f"/*.wav; do
      (
      sourcefile=$file
      filename="${file%.*}"
      destfile="$filename.dst.wav"

      echo $sourcefile;
      ffmpeg -y -i $sourcefile -loglevel error -v quiet -acodec pcm_s16le -ar 4000 -ac 1 $destfile
      ffmpeg -y -i $destfile -loglevel error -v quiet -acodec pcm_s16le -ar 8000 -ac 1 $sourcefile
      rm $destfile
      ) &
    done
    wait
  fi
done

