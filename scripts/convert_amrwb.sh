#!/bin/bash

BITRATE=$2
DIR=$1

for f in "$DIR"/*/*/*; do
  if [ -d "$f" ]; then
    for file in "$f"/*.wav; do
      (
      sourcefile=$file
      filename="${file%.*}"
      destfile="$filename.amr"

      echo $sourcefile;
      ffmpeg -loglevel error -y -i $sourcefile -vn -ar 16000 -ac 1 -b:a $BITRATE -acodec amr_wb $destfile
      ffmpeg -loglevel error -y -i $destfile -v quiet -acodec pcm_s16le -ar 8000 -ac 1 $sourcefile
      rm $destfile
      ) &
    done
    wait
  fi
done


