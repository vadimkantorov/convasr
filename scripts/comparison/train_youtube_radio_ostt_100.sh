python3 train.py $@ \
  --githttp https://github.com/vadimkantorov/convasr/commit/%h \
  --verbose --lang ru \
  --model JasperNetBig \
  --train-batch-size 256 --val-batch-size 128 \
  --scheduler MultiStepLR --decay-milestones 30000 \
  --iterations 35000 \
  --lr 1e-2 \
  --optimizer NovoGrad \
  --train-data-path data/splits/youtube_100h_train.csv.json data/splits/radio_100h_2-4.2sec_train.json \
  --val-data-path data/clean_val.csv.json data/mixed_val.csv.json kontur_calls_micro/kontur_calls_micro.csv.json kontur_calls_micro/kontur_calls_micro.0.csv.json kontur_calls_micro/kontur_calls_micro.1.csv.json data/splits/youtube/cut_microval.json data/splits/youtube_100h_val.csv.json data/kfold_splits/22052020/valset_kfold_22052020.0_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020.1_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020_fold_0.csv.json \
  --analyze kontur_calls_micro.csv \
  --val-iteration-interval 2500 \
  --fp16 O2 \
  --experiment-name youtube_radio_osst_100h \
  --skip-on-epoch-end-evaluation \
  --epochs 105 --exphtml= #\

