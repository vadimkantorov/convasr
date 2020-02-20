IN=$1
OUT=${2:-.}
EXT=${EXT:-%(ext)s}
SUBLANG=${SUBLANG:-ru}

# wget https://yt-dl.org/downloads/latest/youtube-dl && chmod +x ./youtube-dl

./youtube-dl --quiet --no-warnings --sub-lang $SUBLANG --write-sub --write-auto-sub --convert-subs srt --extract-audio -o "$OUT/%(id)s.$EXT" "$IN" --exec echo
