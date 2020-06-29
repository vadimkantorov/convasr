python3 train.py $@ \
  --githttp https://github.com/vadimkantorov/convasr/commit/%h \
  --verbose --lang ru \
  --model JasperNetBig \
  --train-batch-size 64 --val-batch-size 64 \
  --lr 5e-5 \
  --optimizer NovoGrad \
  --train-data-path data/kfold_splits/22052020/trainset_kfold_22052020_fold_0.csv.json \
  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs512____radio_1800h_youtube_finetune/checkpoint_epoch70_iter0241863.pt \
  --val-data-path data/clean_val.csv.json youtube/cut/cut_test.json data/mixed_val.csv.json kontur_calls_micro/kontur_calls_micro.csv.json kontur_calls_micro/kontur_calls_micro.0.csv.json kontur_calls_micro/kontur_calls_micro.1.csv.json echomsk6000/cut/cut_test.json data/kfold_splits/22052020/valset_kfold_22052020.0_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020.1_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020_fold_0.csv.json \
  --val-iteration-interval 5000 \
  --fp16 O2 \
  --experiment-name radio_1800h_youtube_finetune_domain \
  --epochs 150 --exphtml= #\

