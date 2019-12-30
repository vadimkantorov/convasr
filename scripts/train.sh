python3 train.py $@ \
  --githttp https://github.com/vadimkantorov/convasr/commit/%h \
  --verbose --lang ru \
  --model JasperNetBig \
  --train-batch-size 256 --val-batch-size 256 \
  --scheduler MultiStepLR --decay-milestones 100000 175000 \
  --lr 1e-3 \
  --optimizer NovoGrad \
  --train-data-path data/radio_train_0.csv \
  --val-data-path data/radio_val.csv data/mixed_val.csv data/clean_val.csv kontur_calls_micro/kontur_calls_micro.csv kontur_calls_micro/kontur_calls_micro.0.csv kontur_calls_micro/kontur_calls_micro.1.csv \
  --analyze kontur_calls_micro.csv \
  --val-iteration-interval 2500 \
  --experiment-name pretrain_11 \
  --checkpoint data/experiments/radio_domainset_0/checkpoint_epoch04_iter0028700.pt \
  --fp16 O2 \
  --epochs 10 #\

#  --fp16 O2 \
#  --finetune --checkpoint-skip
#  --bpe data/tts_dataset_bpe_5000_word.model \
#  --checkpoint data/checkpoint_epoch02_iter0065000.pt \

#  --bpe data/tts_dataset_tri_1000.model \

#  --train-data-path data/tts_dataset/tts_dataset_train.csv \
#  --train-data-path data/mixed_train.csv data/tts_dataset/tts_dataset_train.csv --train-data-mixing 0.5 0.5 \
#  --train-data-path data/mixed_train.csv \

#  --bpe ../data/spm_train_v05_cleaned_asr_10s_phoneme.model
#  --bpe data/ru.wiki.bpe.vs5000.model
