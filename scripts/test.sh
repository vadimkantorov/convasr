set -e

python3 train.py $@ \
  --align \
  --val-data-path data/mixed_val.csv data/clean_val.csv ../sample_ok/sample_ok.convasr.csv ../sample_ok/sample_ok.convasr.0.csv ../sample_ok/sample_ok.convasr.1.csv data/valset11102019/valset_2019-10-11_0.csv data/valset11102019/valset_2019-10-11_1.csv data/valset11102019/valset_2019-10-11.csv \
  --checkpoint data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0065000.pt #data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0050000.pt data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0052500.pt data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0055000.pt data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0057500.pt data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0060000.pt data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0062500.pt 

#--decoder BeamSearchDecoder --beam-width 5000 --lm  chats_05_prune.binary  #charlm/chats_06_noprune_char.binary # #--lm data/ru_wiyalen_no_punkt.arpa.binary 
#  --val-data-path ../sample_ok/sample_ok.convasr.csv \

#python3 vis.py vis data/logits_sample_ok.convasr.csv.pt

#  --checkpoint data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80___sow/checkpoint_epoch02_iter0060000.pt \

#  --val-data-path ../sample_ok/sample_ok.convasr.csv ../sample_ok/sample_ok.convasr.0.csv ../sample_ok/sample_ok.convasr.1.csv \

    #data/valset11102019/valset_2019-10-11_0.csv data/valset11102019/valset_2019-10-11_1.csv data/valset11102019/valset_2019-10-11.csv \
    #data/mixed_val.csv data/clean_val.csv

#  --checkpoint data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0062500.pt \

#  --val-waveform-transform MixExternalNoise --val-waveform-transform-prob 1 --val-waveform-transform-args data/ru_open_stt_noise_small.csv --val-waveform-transform-debug-dir data/debug_aug \

#  --val-waveform-transform MixExternalNoise --val-waveform-transform-prob 1 --val-waveform-transform-args data/sample_ok.noise.csv --val-waveform-transform-debug-dir data/debug_aug \

#  --val-waveform-transform SOXAWN --val-waveform-transform-debug-dir data/debug_aug \

#  data/adapted.csv 
#  --val-batch-size 40 --val-data-path  ../open_stt_splits/splits/clean_val.csv ../open_stt_splits/splits/mixed_val.csv ../sample_ok/sample_ok.convasr.csv \
#  --checkpoint model_checkpoint_0027_epoch_02.model.pt \
#  --noise-data-path data/ru_open_stt_noise_small.csv --noise-level 0.5 \
