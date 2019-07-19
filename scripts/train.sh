CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
  --lang ru \
  --lr 5e-3 --momentum 0.5 --weight-decay 1e-5 --nesterov \
  --train-data-path ../open_stt_splits/splits/clean_train.csv.gz \
  --val-data-path ../sample_ok/sample_ok.convasr.csv \
  --data-parallel \
  --epochs 30 

