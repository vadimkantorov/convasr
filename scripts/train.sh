CUDA_VISIBLE_DEVICES=0,1 python3 train.py $@ \
  --verbose --lang ru \
  --model JasperNet \
  --train-batch-size 80 --val-batch-size 64 \
  --lr 1e-3 \
  --optimizer NovoGrad \
  --train-data-path data/mixed_train.csv \
  --val-data-path data/mixed_val.csv data/clean_val.csv ../sample_ok/sample_ok.convasr.csv ../sample_ok/sample_ok.convasr.0.csv ../sample_ok/sample_ok.convasr.1.csv data/tts_dataset/tts_dataset_val.csv \
  --val-iteration-interval 2500 \
  --epochs 5 \
  --dropout 0 --weight-decay 0 \
  --checkpoint data/experiments/JasperNet_NovoGrad_lr1e-2_wd0e+00_bs80___separable/checkpoint_epoch01_iter0025000.pt

#  --scheduler MultiStepLR --decay-milestones 25000 \

#  --train-data-path data/tts_dataset/tts_dataset_train.csv \
#  --train-data-path data/mixed_train.csv data/tts_dataset/tts_dataset_train.csv --train-data-mixing 0.5 0.5 \
# --train-data-path data/mixed_train.csv \

#  --bpe ../data/spm_train_v05_cleaned_asr_10s_phoneme.model
#  --bpe data/ru.wiki.bpe.vs5000.model
