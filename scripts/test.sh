CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
  --lang ru \
  --checkpoint data/checkpoints/checkpoint_epoch00_iter0020000.pt \
  --val-data-path ../sample_ok/sample_ok.convasr.csv

#  --checkpoint model_checkpoint_0027_epoch_02.model.pt \
