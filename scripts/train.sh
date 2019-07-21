CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
  --lang ru \
  --lr 5e-3 --momentum 0.5 --weight-decay 1e-5 --nesterov \
  --train-data-path ../open_stt_splits/splits/mixed_train.csv \
  --val-data-path ../open_stt_splits/splits/clean_val.csv ../open_stt_splits/splits/mixed_val.csv ../sample_ok/sample_ok.convasr.csv \
  --epochs 1 

