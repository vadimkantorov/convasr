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
      ffmpeg -loglevel panic -y -i $sourcefile -vn -ar 8000 -ac 1 -b:a $BITRATE -acodec amr_nb $destfile
      ffmpeg -loglevel panic -y -i $destfile -loglevel error -v quiet -acodec pcm_s16le -ar 8000 -ac 1 $sourcefile
      rm $destfile
      ) &
    done
    wait
  fi
done
