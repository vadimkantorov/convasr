#python3 diarization.py ref $@ -i diarization/stereo -o data/diarization/ref --html

#python3 diarization.py hyp $@ -i diarization/mono -o data/diarization/hyp --html

python3 diarization.py eval --ref data/diarization/ref --hyp data/diarization/hyp #--audio

#python3 diarization.py ref $@ -i diarization/mono/00d13c16-ac0d-409c-8a5e-36741a9e750a.mp3.wav -o data/diarization 

#python3 diarization.py ref $@ -i diarization/mono/0a8b743b-06b6-4a71-8d47-fd713dee039f.mp3.wav -o data/diarization 

#python3 diarization.py -i b70a59ea-5caa-4783-9bb7-32b4eceec16a.mp3.wav -o data/diarization.json
#python3 vis.py transcript --mono -i data/diarization.json -o data/diarization.html
