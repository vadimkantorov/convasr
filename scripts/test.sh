CUDA_VISIBLE_DEVICES=0,1 python3 train.py $@ \
  --lang ru \
  --model Wav2LetterRu \
  --checkpoint "data/experiments/Wav2LetterRu_SGD_lr1e-2_wd1e-3_bs80_trainWT_MixExternalNoise(noise_level=1.0,noise_data_path='data_sample_ok.noise.csv')_valWT_MixExternalNoise(noise_level=1.0,noise_data_path='data_sample_ok.noise.csv')/checkpoint_epoch00_iter0010000.pt" \
  --val-batch-size 64 --val-data-path data/mixed_val.csv ../sample_ok/sample_ok.convasr.csv

#  --val-waveform-transforms 'AddWhiteNoise(0.025)' \

#data/experiments/Wav2LetterRu_SGD_lr1e-02_wd1e-05_bs80/checkpoint_epoch00_iter0017500.pt \
#  --val-waveform-transforms 'MixExternalNoise("data/sample_ok.noise.csv", 1.0)' \
#  --val-batch-size 40 --val-data-path  ../open_stt_splits/splits/clean_val.csv ../open_stt_splits/splits/mixed_val.csv ../sample_ok/sample_ok.convasr.csv \
#  --checkpoint model_checkpoint_0027_epoch_02.model.pt \
#  --noise-data-path data/ru_open_stt_noise_small.csv --noise-level 0.5 \
