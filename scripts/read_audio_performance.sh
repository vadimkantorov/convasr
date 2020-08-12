# generate test files
# ffmpeg -f lavfi -i "sine=frequency=1000:duration=3600" -c:a pcm_s16le -ar 8000 data/tests/test_1h.wav
# ffmpeg -f lavfi -i "sine=frequency=1000:duration=60" -c:a pcm_s16le -ar 8000 data/tests/test_1m.wav
# ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" -c:a pcm_s16le -ar 8000 data/tests/test_5s.wav

for backend in sox ffmpeg soundfile scipy; do
  for file in test_5s.wav test_1m.wav test_1h.wav; do
    python audio.py timeit --audio-path data/tests/$file --mono --audio-backend $backend
  done
done
