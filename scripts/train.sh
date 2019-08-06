CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
  --verbose --lang ru \
  --model Wav2LetterRu \
  --train-batch-size 40 --val-batch-size 80 \
  --lr 1e-2 --optimizer SGD \
  --train-data-path /root/convasr/data/clean_train.csv \
  --val-data-path /root/convasr/data/clean_val.csv \
  --val-iteration-interval 10 \
  --epochs 1 

#  --train-data-path ../open_stt_splits/splits/mixed_train.csv \
#  --val-data-path ../open_stt_splits/splits/clean_val.csv ../sample_ok/sample_ok.convasr.csv \
