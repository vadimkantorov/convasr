python3 train.py $@ \
  --githttp https://github.com/vadimkantorov/convasr/commit/%h \
  --verbose --lang ru \
  --model JasperNetBig \
  --train-batch-size 1536 --val-batch-size 512 \
  --scheduler MultiStepLR --decay-milestones 300000 400000 \
  --iterations 600000 \
  --lr 1e-2 \
  --optimizer NovoGrad \
  --train-data-path youtube/cut/cut_train.json echomsk6000/cut2/cut2_train.json \
  --val-data-path data/clean_val.csv.json data/mixed_val.csv.json kontur_calls_micro/kontur_calls_micro.csv.json kontur_calls_micro/kontur_calls_micro.0.csv.json kontur_calls_micro/kontur_calls_micro.1.csv.json youtube/cut/cut_microval.json data/kfold_splits/22052020/valset_kfold_22052020.0_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020.1_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020_fold_0.csv.json \
  --val-iteration-interval 5000 \
  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs1536____youtube_3000h_better_pt/checkpoint_epoch242_iter0347896.pt \
  --fp16 O2 \
  --experiment-name youtube_3000h_better_pt \
  --epochs 1000 --exphtml= #\

#data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs1536____youtube_3000h/checkpoint_epoch236_iter0335638.pt
#data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs1536____youtube_3000h/checkpoint_epoch240_iter0345000.pt


