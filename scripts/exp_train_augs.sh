python3 train.py \
  --lang ru  --model Wav2LetterRu \
  --train-batch-size 80 --val-batch-size 80 \
  --lr 1e-2 --weight-decay 1e-3 --optimizer SGD \
  --scheduler MultiStepLR --decay-milestones 25000 40000 50000 \
  --train-data-path data/mixed_train.csv \
  --val-data-path data/mixed_val.csv data/clean_val.csv data/calls_val.csv \
  --sample-rate 8000 \
  --epochs 3 \
  --train-waveform-transform PS --train-waveform-transform-prob 0.0 \
  --name PS

#python3 train.py \
#  --lang ru  --model Wav2LetterRu \
#  --train-batch-size 80 --val-batch-size 80 \
#  --lr 1e-2 --weight-decay 1e-3 --optimizer SGD \
#  --scheduler MultiStepLR --decay-milestones 25000 40000 50000 \
#  --train-data-path data/mixed_train.csv \
#  --val-data-path data/mixed_val.csv data/clean_val.csv data/calls_val.csv \
#  --sample-rate 8000 \
#  --epochs 2 \
#  --train-waveform-transform SP --train-waveform-transform-prob 0.5 \
#  --name SP
#
#python3 train.py \
#  --lang ru  --model Wav2LetterRu \
#  --train-batch-size 80 --val-batch-size 80 \
#  --lr 1e-2 --weight-decay 1e-3 --optimizer SGD \
#  --scheduler MultiStepLR --decay-milestones 25000 40000 50000 \
#  --train-data-path data/mixed_train.csv \
#  --val-data-path data/mixed_val.csv data/clean_val.csv data/calls_val.csv \
#  --sample-rate 8000 \
#  --epochs 2 \
#  --train-waveform-transform AMRNB --train-waveform-transform-prob 0.5 \
#  --name AMRNB
#
#python3 train.py \
#  --lang ru  --model Wav2LetterRu \
#  --train-batch-size 80 --val-batch-size 80 \
#  --lr 1e-2 --weight-decay 1e-3 --optimizer SGD \
#  --scheduler MultiStepLR --decay-milestones 25000 40000 50000 \
#  --train-data-path data/mixed_train.csv \
#  --val-data-path data/mixed_val.csv data/clean_val.csv data/calls_val.csv \
#  --sample-rate 8000 \
#  --epochs 3 \
#  --train-waveform-transform PSSPAMRNB --train-waveform-transform-prob 0.7 \
#  --name PSSPAMRNB
