CUDA_VISIBLE_DEVICES=0,1 python3 train.py --verbose --lang ru \
  --model Wav2LetterRu \
  --train-batch-size 40 --val-batch-size 80 \
  --lr 5e-3 --optimizer AdamW \
  --train-data-path ../open_stt_splits/splits/mixed_train.csv \
  --val-batch-period 5000 \
  --val-data-path ../open_stt_splits/splits/clean_val.csv ../sample_ok/sample_ok.convasr.csv \
  --epochs 1 

#  --val-data-path ../open_stt_splits/splits/clean_val.csv ../open_stt_splits/splits/mixed_val.csv ../sample_ok/sample_ok.convasr.csv \
