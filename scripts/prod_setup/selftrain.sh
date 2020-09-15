python3 train.py $@ \
  --githttp https://github.com/vadimkantorov/convasr/commit/%h \
  --verbose --lang ru \
  --model JasperNetBig \
  --train-batch-size 1536 --val-batch-size 512 \
  --lr 1e-4 \
  --optimizer NovoGrad \
  --train-data-path unsup_dataset/unsup_dataset_transcripts_simple_1608_morph.csv.json \
  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs1536____longtrain_self_train/checkpoint_epoch335_iter0695000.pt \
  --val-data-path data/clean_val.csv.json data/mixed_val.csv.json kontur_calls_micro/kontur_calls_micro.csv.json kontur_calls_micro/kontur_calls_micro.0.csv.json kontur_calls_micro/kontur_calls_micro.1.csv.json youtube/cut/cut_microval.json domain_set/transcripts/16082020/valset_16082020.0.csv.json domain_set/transcripts/16082020/valset_16082020.1.csv.json \
  --val-iteration-interval 5000 \
  --fp16 O2 \
  --iterations 755000 \
  --experiment-name longtrain_self_train \
  --window-size 0.04 \
  --skip-on-epoch-end-evaluation \
  --epochs 1000 --exphtml= #\

# init cp data/experiments/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs1536____summary_youtube_ws_additional_radio_lr001/checkpoint_epoch311_iter0645000.pt
