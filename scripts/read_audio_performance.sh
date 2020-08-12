# generate test files
# ffmpeg -f lavfi -i "sine=frequency=1000:duration=3600" -c:a pcm_s16le -ar 8000 data/tests/test_1h.wav
# ffmpeg -f lavfi -i "sine=frequency=1000:duration=60" -c:a pcm_s16le -ar 8000 data/tests/test_1m.wav
# ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" -c:a pcm_s16le -ar 8000 data/tests/test_5s.wav

# wav
python audio.py timeit --audio-path data/tests/test_5s.wav --mono --audio-backend soundfile --numbers 100
python audio.py timeit --audio-path data/tests/test_1m.wav --mono --audio-backend soundfile --numbers 100
python audio.py timeit --audio-path data/tests/test_1h.wav --mono --audio-backend soundfile --numbers 100

python audio.py timeit --audio-path data/tests/test_5s.wav --mono --audio-backend ffmpeg --numbers 100
python audio.py timeit --audio-path data/tests/test_1m.wav --mono --audio-backend ffmpeg --numbers 100
python audio.py timeit --audio-path data/tests/test_1h.wav --mono --audio-backend ffmpeg --numbers 100

python audio.py timeit --audio-path data/tests/test_5s.wav --mono --audio-backend sox --numbers 100
python audio.py timeit --audio-path data/tests/test_1m.wav --mono --audio-backend sox --numbers 100
python audio.py timeit --audio-path data/tests/test_1h.wav --mono --audio-backend sox --numbers 100
