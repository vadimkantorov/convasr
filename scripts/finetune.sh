python3 train.py $@ \
  --githttp https://github.com/vadimkantorov/convasr/commit/%h \
  --verbose --lang ru \
  --model JasperNetBig \
  --train-batch-size 8 --val-batch-size 256 \
  --scheduler MultiStepLR --decay-milestones 25000 35000 \
  --lr 1e-3 \
  --optimizer NovoGrad \
  --train-data-path data/trainset_by_rec22122019.0.csv \
  --val-data-path data/mixed_val.csv data/clean_val.csv data/valset_by_rec22122019.0.csv data/valset_by_rec22122019.0.csv data/valset_by_rec22122019.1.csv data/valset_by_rec22122019.csv \
  --val-iteration-interval 2500 \
  --epochs 30 \
  --bpe data/tts_dataset_bi_200.model \
  --experiment-name bpe_bi200_conv1d_epilogue40_finetune_decays_newval_freeze7_lr001_bs8_by_rec_ch0 \
  --bpe data/tts_dataset_bi_200.model \
  --fp16 O2 \
  --checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256___bpe_bpe_bi200_conv1d_epilogue40_checkpoint_epoch06_iter0050498.pt \
  --freeze_firstn_layers 7

#  --analyze kontur_calls_micro.csv \
#  --loads_only_firstn_layers 10 \
