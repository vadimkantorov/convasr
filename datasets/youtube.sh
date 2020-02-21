IN=$1
OUT=${2:-.}
EXT=${EXT:-%(ext)s}
SUBLANG=${SUBLANG:-ru}

# wget https://yt-dl.org/downloads/latest/youtube-dl && chmod +x ./youtube-dl

./youtube-dl --quiet --no-warnings --write-info-json --sub-lang $SUBLANG --write-sub --write-auto-sub --convert-subs srt --extract-audio -o "$OUT/%(id)s.$EXT" "$IN" --exec echo

./youtube-dl --get-id --flat-playlist $IN  | sed 's/^/http:\/\/youtu.be\//'
