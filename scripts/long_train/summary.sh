python3 train.py $@ \
  --githttp https://github.com/vadimkantorov/convasr/commit/%h \
  --verbose --lang ru \
  --model JasperNetBig \
  --train-batch-size 1536 --val-batch-size 512 \
  --iterations 600000 \
  --lr 1e-2 \
  --optimizer NovoGrad \
  --train-data-path youtube/cut/cut_train.json youtube_lowpass/cut/cut_train.json youtube_gsmfr/cut/cut_train.json youtube_amrnb12200/cut/cut_train.json echomsk6000/cut2/cut2_train.json \
  --val-data-path data/clean_val.csv.json data/mixed_val.csv.json kontur_calls_micro/kontur_calls_micro.csv.json kontur_calls_micro/kontur_calls_micro.0.csv.json kontur_calls_micro/kontur_calls_micro.1.csv.json youtube/cut/cut_microval.json data/kfold_splits/22052020/valset_kfold_22052020.0_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020.1_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020_fold_0.csv.json \
  --val-iteration-interval 1000 \
  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs1536____summary/checkpoint_epoch259_iter0388000.pt \
  --fp16 O2 \
  --experiment-name summary \
  --window-size 0.04 \
  --epochs 1000 --exphtml= #\


#--scheduler MultiStepLR --decay-milestones 300000 400000 \
#echomsk6000/cut2/cut2_train.json