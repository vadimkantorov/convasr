python3 train.py $@ \
  --githttp https://github.com/vadimkantorov/convasr/commit/%h \
  --verbose --lang ru \
  --model JasperNetBig \
  --train-batch-size 1024 --val-batch-size 512 \
  --lr 1e-4 \
  --optimizer NovoGrad \
  --train-data-path data/kfold_splits/22052020/trainset_kfold_22052020_fold_0.csv.json \
  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs256____youtube_3000h_finetune_2/checkpoint_epoch448_iter0435000.pt \
  --val-data-path data/clean_val.csv.json data/mixed_val.csv.json kontur_calls_micro/kontur_calls_micro.csv.json kontur_calls_micro/kontur_calls_micro.0.csv.json kontur_calls_micro/kontur_calls_micro.1.csv.json youtube/cut/cut_microval.json data/kfold_splits/22052020/valset_kfold_22052020.0_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020.1_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020_fold_0.csv.json \
  --val-iteration-interval 1000 \
  --fp16 O2 \
  --experiment-name youtube_3000h_finetune_2 \
  --skip-on-epoch-end-evaluation \
  --epochs 1000 --exphtml= #\


