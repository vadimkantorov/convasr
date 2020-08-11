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
      rm $sourcefile
      ) &
    done
    wait
  fi
done
