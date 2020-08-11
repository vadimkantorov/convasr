python3 train.py $@ \
  --githttp https://github.com/vadimkantorov/convasr/commit/%h \
  --verbose --lang ru \
  --model JasperNetBig \
  --train-batch-size 512 --val-batch-size 256 \
  --lr 1e-5 \
  --optimizer NovoGrad \
  --train-data-path data/kfold_splits/22052020/trainset_kfold_22052020_fold_0.csv.json \
  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs512____youtube_3000h_after_selftrain_2try_middle_morphy/checkpoint_epoch371_iter0477000.pt \
  --val-data-path data/clean_val.csv.json data/mixed_val.csv.json kontur_calls_micro/kontur_calls_micro.csv.json kontur_calls_micro/kontur_calls_micro.0.csv.json kontur_calls_micro/kontur_calls_micro.1.csv.json youtube/cut/cut_microval.json data/kfold_splits/22052020/valset_kfold_22052020.0_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020.1_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020_fold_0.csv.json \
  --val-iteration-interval 1000 \
  --iterations 481000 \
  --fp16 O2 \
  --experiment-name youtube_3000h_after_selftrain_2try_middle_morphy \
  --skip-on-epoch-end-evaluation \
  --epochs 1000 --exphtml= #\


#init data/experiments/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs512____youtube_3000h_selftrain_2try_middle_morphy/checkpoint_epoch317_iter0470000.pt

# init pt data/experiments/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs1536____youtube_3000h_better_pt/checkpoint_epoch261_iter0386713.pt
