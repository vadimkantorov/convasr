CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
  --verbose --lang ru \
  --model Wav2LetterRu \
  --train-batch-size 80 --val-batch-size 80 \
  --lr 1e-2 --optimizer SGD \
  --train-data-path /root/convasr/data/mixed_train.csv \
  --val-data-path /root/convasr/data/mixed_val.csv \
  --val-iteration-interval 1000 \
  --epochs 5 

#  --train-data-path ../open_stt_splits/splits/mixed_train.csv \
#  --val-data-path ../open_stt_splits/splits/clean_val.csv ../sample_ok/sample_ok.convasr.csv \
