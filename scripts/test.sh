CUDA_VISIBLE_DEVICES=0,1 python3 train.py $@ \
  --lang ru --model Wav2LetterRu \
  --checkpoint data/checkpoint_epoch04_iter0124135.pt \
  --val-waveform-transform SOXGP --val-waveform-transform-debug-dir data/debug_aug \
  --val-batch-size 64 --val-data-path data/clean_val.csv #../sample_ok/sample_ok.convasr.csv

#  --val-waveform-transform AddWhiteNoise --val-waveform-transform-prob nan \

#  --val-waveform-transform MixExternalNoise --val-waveform-transform-prob 1 --val-waveform-transform-args data/sample_ok.noise.csv \

#  --decoder BeamSearchDecoder --beam-width 5000 --lm chats.binary #data/ru_wiyalen_no_punkt.arpa.binary


#  data/adapted.csv 
#  --val-batch-size 40 --val-data-path  ../open_stt_splits/splits/clean_val.csv ../open_stt_splits/splits/mixed_val.csv ../sample_ok/sample_ok.convasr.csv \
#  --checkpoint model_checkpoint_0027_epoch_02.model.pt \
#  --noise-data-path data/ru_open_stt_noise_small.csv --noise-level 0.5 \
