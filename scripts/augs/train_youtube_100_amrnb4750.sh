python3 train.py $@ \
  --githttp https://github.com/vadimkantorov/convasr/commit/%h \
  --verbose --lang ru \
  --model JasperNetBig \
  --train-batch-size 256 --val-batch-size 128 \
  --iterations 45000 \
  --lr 1e-3 \
  --optimizer NovoGrad \
  --train-data-path youtube_amrnb_475/cut/100h_train/cut_train_100h.json \
  --val-data-path youtube_amrnb_475/cut/100h_train/cut_train_100h_val.json data/clean_val.csv.json data/mixed_val.csv.json kontur_calls_micro/kontur_calls_micro.csv.json kontur_calls_micro/kontur_calls_micro.0.csv.json kontur_calls_micro/kontur_calls_micro.1.csv.json data/splits/youtube/cut_microval.json data/splits/youtube_100h_val.csv.json data/kfold_splits/22052020/valset_kfold_22052020.0_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020.1_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020_fold_0.csv.json \
  --analyze kontur_calls_micro.csv \
  --val-iteration-interval 1000 \
  --fp16 O2 \
  --experiment-name amrnb \
  --skip-on-epoch-end-evaluation \
  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____exp_youtube_100h/checkpoint_epoch89_iter0035000.pt \
  --epochs 305 --exphtml= #\

