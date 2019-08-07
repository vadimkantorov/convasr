CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
  --verbose --lang ru \
  --model Wav2LetterRu \
  --train-batch-size 80 --val-batch-size 80 \
  --lr 2e-2 --optimizer SGD \
  --train-data-path ../open_stt_splits/splits/mixed_train.csv \
  --val-data-path ../open_stt_splits/splits/clean_val.csv ../open_stt_splits/splits/mixed_val.csv ../sample_ok/sample_ok.convasr.csv \
  --val-iteration-interval 2500 \
  --scheduler MultiStepLR --milestones 10000 --gamma 0.1 \
  --epochs 1 

#  --scheduler PolynomialDecayLR --scheduler-decay-epochs 1 --lr-end 1e-5 \
#  --train-data-path /root/convasr/data/mixed_train.csv \
#  --val-data-path /root/convasr/data/mixed_val.csv \
