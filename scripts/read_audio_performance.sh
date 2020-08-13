set -e
# generate test files
#mkdir -p data/tests
#ffmpeg -y -f lavfi -i "sine=frequency=1000:duration=3600" -c:a pcm_s16le -ar 8000 data/tests/test_1h.wav
#ffmpeg -y -f lavfi -i "sine=frequency=1000:duration=60" -c:a pcm_s16le -ar 8000 data/tests/test_1m.wav
#ffmpeg -y -f lavfi -i "sine=frequency=1000:duration=5" -c:a pcm_s16le -ar 8000 data/tests/test_5s.wav

echo "| file       | reads count |  backend | process_time us| perf_counter us|"
echo "|-----------:|---:|---------:|--------------:|-------------:|"
for backend in scipy soundfile sox ffmpeg; do
  for file in test_5s.wav test_1m.wav test_1h.wav; do
    python audio.py timeit --audio-path data/tests/$file --mono --audio-backend $backend
  done
done
