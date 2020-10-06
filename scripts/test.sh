set -e

python3 train.py $@ \
  --analyze \
  --device cpu \
  --checkpoint data/speechcore/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs512____rerun_finetune_after_self_train_epoch183_iter0290000.pt \
  --val-data-path kontur_calls_micro/kontur_calls_micro.csv.json # data/speechcore/best_checkpoint_and_valset_2508/valset_16082020.csv.json --frontend-in-model 

# --checkpoint data/speechcore/best_checkpoint_and_valset_2508/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs512____longtrain_finetune_after_self_train_checkpoint_epoch400_iter0740000.pt \

#  --fp16 O2 \

#python3 train.py $@ \
#  --exphtml= --githttp https://github.com/vadimkantorov/convasr/commit/%h \
#  --analyze \
#  --checkpoint data/speechcore/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs1024____long-train_bs1024_step6_430k_checkpoint_epoch259_iter0468272.pt \
#  --val-data-path kontur_calls_micro/kontur_calls_micro.csv.json --frontend-in-model #--batch-time-padding-multiple 1

#--checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____fp16O2/checkpoint_epoch05_iter0040000.pt \

#  --decoder BeamSearchDecoder --lm data/speechcore/kenlm_models/soft_prune.bin

#python3 vis.py errors data/transcripts_kontur_calls_micro.csv.json_BeamSearchDecoder_hard.json --theirs data/transcripts_kontur_calls_micro.csv.json_BeamSearchDecoder_soft.json

 # kontur_calls_micro/kontur_calls_micro.csv #valset17122019/valset17122019.csv # valset17122019/valset_by_rec22122019.csv #data/clean_val.csv data/mixed_val.csv  valset11102019/valset11102019.csv #valset17122019/valset17122019.csv # kontur_calls_micro/kontur_calls_micro.0.csv kontur_calls_micro/kontur_calls_micro.1.csv data/clean_val.csv data/ data/valset17122019/valset17122019.0.csv data/valset17122019/valset17122019.1.csv \

#python3 vis.py logits data/logits_kontur_calls_micro.csv.pt

#  --decoder BeamSearchDecoder --beam-width 1000 --decoder-topk 1000 --lm  data/lm/chats_05_prune.binary \

#  --val-data-path data/clean_val.csv kontur_calls_micro/kontur_calls_micro.csv kontur_calls_micro/kontur_calls_micro.0.csv kontur_calls_micro/kontur_calls_micro.1.csv  data/valset11102019/valset11102019.0.csv data/valset11102019/valset11102019.1.csv

#--checkpoint data/checkpoint_epoch02_iter0065000.pt #data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0050000.pt data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0052500.pt data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0055000.pt data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0057500.pt data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0060000.pt data/experiments/JasperNetBigInplace_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0062500.pt 


#  --bpe data/tts_dataset_bpe_1000.model \
#  --val-data-path ../sample_ok/sample_ok.convasr.csv ../sample_ok/sample_ok.convasr.0.csv ../sample_ok/sample_ok.convasr.1.csv data/valset11102019/valset_2019-10-11_0.csv data/valset11102019/valset_2019-10-11_1.csv data/valset11102019/valset_2019-10-11.csv \



#  --val-data-path ../sample_ok/sample_ok.convasr.csv ../sample_ok/sample_ok.convasr.0.csv ../sample_ok/sample_ok.convasr.1.csv \

    #data/valset11102019/valset_2019-10-11_0.csv data/valset11102019/valset_2019-10-11_1.csv data/valset11102019/valset_2019-10-11.csv \
    # data/clean_val.csv

#  --val-waveform-transform SpecLowPass 3500

#--val-waveform-transform-prob 1 --val-waveform-transform-args data/ru_open_stt_noise_small.csv --val-waveform-transform-debug-dir data/debug_aug \

#  --val-waveform-transform MixExternalNoise --val-waveform-transform-prob 1 --val-waveform-transform-args data/sample_ok.noise.csv --val-waveform-transform-debug-dir data/debug_aug \

#  --val-waveform-transform SOXAWN --val-waveform-transform-debug-dir data/debug_aug \

#  data/adapted.csv 
#  --val-batch-size 40 --val-data-path  ../open_stt_splits/splits/clean_val.csv ../open_stt_splits/splits/mixed_val.csv ../sample_ok/sample_ok.convasr.csv \
#  --noise-data-path data/ru_open_stt_noise_small.csv --noise-level 0.5 \
