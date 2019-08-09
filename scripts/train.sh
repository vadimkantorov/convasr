CUDA_VISIBLE_DEVICES=0,1 python3 train.py $@ \
  --verbose --lang ru \
  --model Wav2LetterRu \
  --train-batch-size 80 --val-batch-size 80 \
  --lr 1e-2 --weight-decay 1e-3 --optimizer SGD \
  --train-data-path data/mixed_train.csv \
  --val-data-path data/mixed_val.csv ../sample_ok/sample_ok.convasr.csv \
  --val-iteration-interval 2500 \
  --scheduler PolynomialDecayLR --decay-epochs 3 --lr-end 1e-4 \
#  --noise-data-path data/ru_open_stt_noise_small.csv --noise-level 0.7 \
  --epochs 1 

#  --scheduler MultiStepLR --decay-milestones 10000 30000 40000 80000 --decay-gamma 0.5 \
#  --val-data-path ../open_stt_splits/splits/clean_val.csv ../open_stt_splits/splits/mixed_val.csv ../sample_ok/sample_ok.convasr.csv \
#  --scheduler PolynomialDecayLR --scheduler-decay-epochs 1 --lr-end 1e-5 \
#  --train-data-path /root/convasr/data/mixed_train.csv \
#  --val-data-path /root/convasr/data/mixed_val.csv \
