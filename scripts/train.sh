python3 train.py $@ \
  --verbose --lang ru \
  --model JasperNetBig \
  --train-batch-size 4 --val-batch-size 64 \
  --scheduler MultiStepLR --decay-milestones 25000 75000 \
  --lr 1e-2 \
  --optimizer NovoGrad \
  --train-data-path data/mixed_train.csv.json \
  --val-iteration-interval 2500 \
  --val-data-path kontur_calls_micro/kontur_calls_micro.csv.json kontur_calls_micro/kontur_calls_micro.0.csv.json kontur_calls_micro/kontur_calls_micro.1.csv.json \
  --epochs 3 --name debug --checkpoint-skip --iterations 10 
# data/mixed_val.csv data/clean_val.csv 
#  --val-data-path data/mixed_val.csv data/clean_val.csv kontur_calls_micro/kontur_calls_micro.csv kontur_calls_micro/kontur_calls_micro.0.csv kontur_calls_micro/kontur_calls_micro.1.csv \

#  --finetune --checkpoint-skip
#  --bpe data/tts_dataset_bpe_5000_word.model \
#  --checkpoint data/checkpoint_epoch02_iter0065000.pt \

#  --bpe data/tts_dataset_tri_1000.model \

#  --train-data-path data/tts_dataset/tts_dataset_train.csv \
#  --train-data-path data/mixed_train.csv data/tts_dataset/tts_dataset_train.csv --train-data-mixing 0.5 0.5 \
#  --train-data-path data/mixed_train.csv \

#  --bpe ../data/spm_train_v05_cleaned_asr_10s_phoneme.model
#  --bpe data/ru.wiki.bpe.vs5000.model
