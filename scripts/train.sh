CUDA_VISIBLE_DEVICES=0,1 python3 train.py $@ \
  --verbose --lang ru \
  --model JasperNet \
  --train-batch-size 80 --val-batch-size 80 \
  --lr 1e-2 --weight-decay 1e-3 \
  --scheduler MultiStepLR --decay-milestones 25000 \
  --optimizer NovoGrad \
  --train-data-path data/mixed_train.csv \
  --val-data-path data/mixed_val.csv data/clean_val.csv ../sample_ok/sample_ok.convasr.csv ../sample_ok/sample_ok.convasr.0.csv ../sample_ok/sample_ok.convasr.1.csv \
  --val-iteration-interval 2500 \
  --epochs 5 \
  --dropout 0 \
  --checkpoint data/experiments/JasperNet_NovoGrad_lr1e-2_wd1e-3_bs80___testabn2/checkpoint_epoch01_iter0046172.pt

#  --bpe ../data/spm_train_v05_cleaned_asr_10s_phoneme.model
#  --bpe data/ru.wiki.bpe.vs5000.model
