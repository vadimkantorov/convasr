CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
  --verbose --lang ru \
  --model Wav2LetterRu \
  --train-batch-size 80 --val-batch-size 80 \
  --lr 1e-2 --optimizer SGD \
  --train-data-path ../open_stt_splits/splits/mixed_train.csv \
  --val-data-path ../open_stt_splits/splits/clean_val.csv ../sample_ok/sample_ok.convasr.csv \
  --val-iteration-interval 2500 \
  --epochs 1 

#  --val-data-path ../open_stt_splits/splits/clean_val.csv ../open_stt_splits/splits/mixed_val.csv ../sample_ok/sample_ok.convasr.csv \
