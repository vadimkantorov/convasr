CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
  --lang ru --val-batch-size 40 \
  --model JasperNetDenseResidual \
  --checkpoint data/checkpoints_jasper/checkpoint_epoch09_iter0006500.pt \
  --val-data-path ../sample_ok/sample_ok.convasr.csv

#  --checkpoint model_checkpoint_0027_epoch_02.model.pt \
