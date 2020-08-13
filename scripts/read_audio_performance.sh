set -e
# generate test files
#DATA_PATH=data/tests
#mkdir -p $DATA_PATH

#ffmpeg -y -f lavfi -i "sine=frequency=1000:duration=3600" -c:a pcm_s16le -ar 8000 $DATA_PATH/test_1h.wav
#ffmpeg -y -f lavfi -i "sine=frequency=1000:duration=60" -c:a pcm_s16le -ar 8000 $DATA_PATH/test_1m.wav
#ffmpeg -y -f lavfi -i "sine=frequency=1000:duration=5" -c:a pcm_s16le -ar 8000 $DATA_PATH/test_5s.wav

#for file in test_5s test_1m test_1h ; do
#  ffmpeg -y -i $DATA_PATH/$file.wav -c:a libgsm -vn -ar 8000 -ac 1 -ab 13000 -f gsm $DATA_PATH/$file.gsm
#  ffmpeg -y -i $DATA_PATH/$file.wav -c:a libopus -vn -ar 8000 -ac 1 -ab 128000 -f opus $DATA_PATH/$file.opus
#  ffmpeg -y -i $DATA_PATH/$file.wav -c:a libmp3lame -vn -ar 8000 -ac 1 -ab 128000 -f mp3 $DATA_PATH/$file.mp3
#done

echo "| file       | reads count |  backend | process_time us| perf_counter us|"
echo "|-----------:|---:|---------:|--------------:|-------------:|"
for backend in scipy soundfile sox ffmpeg; do
  for file in test_5s.wav test_1m.wav test_1h.wav \
   test_5s.mp3 test_1m.mp3 test_1h.mp3 \
   test_5s.opus test_1m.opus test_1h.opus \
   test_5s.gsm test_1m.gsm test_1h.gsm; do
     python audio.py timeit --audio-path $DATA_PATH/$file --mono --audio-backend $backend
  done
done
