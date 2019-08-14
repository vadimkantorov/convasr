CUDA_VISIBLE_DEVICES=0,1 python3 train.py $@ \
  --lang ru \
  --model Wav2LetterRu \
  --checkpoint "data/experiments/Wav2LetterRu_SGD_lr1e-2_wd1e-3_bs80_augAWNSPGPPS0.3/checkpoint_epoch00_iter0002500.pt" \
  --val-batch-size 64 --val-data-path data/clean_val.csv  ../sample_ok/sample_ok.convasr.csv \
  --decoder beam --beam-width 2048 --lm-path chats.binary #data/ru_wiyalen_no_punkt.arpa.binary

#  --checkpoint data/experiments/Wav2LetterRu_SGD_lr1e-2_wd1e-3_bs80_augAWNSPGPPS0.3_weightnorm/checkpoint_epoch00_iter0002500.pt \
#  data/adapted.csv 
#  --val-waveform-transforms 'MixExternalNoise("data/sample_ok.noise.csv", 1.0)' \
#  --val-batch-size 40 --val-data-path  ../open_stt_splits/splits/clean_val.csv ../open_stt_splits/splits/mixed_val.csv ../sample_ok/sample_ok.convasr.csv \
#  --checkpoint model_checkpoint_0027_epoch_02.model.pt \
#  --noise-data-path data/ru_open_stt_noise_small.csv --noise-level 0.5 \
