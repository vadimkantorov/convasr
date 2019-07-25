CUDA_VISIBLE_DEVICES=0,1 python3 train.py --lang ru \
  --model Wav2LetterRu \
  --checkpoint data/checkpoints_wav2letterru/checkpoint_epoch09_iter0006500.pt \
  --val-batch-size 40 --val-data-path ../sample_ok/sample_ok.convasr.csv

#  --checkpoint model_checkpoint_0027_epoch_02.model.pt \
