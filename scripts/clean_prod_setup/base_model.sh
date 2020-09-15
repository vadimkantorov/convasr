python3 train.py $@ \
  --githttp https://github.com/vadimkantorov/convasr/commit/%h \
  --verbose --lang ru \
  --model JasperNetBig \
  --train-batch-size 1536 --val-batch-size 512 \
  --iterations 230000 \
  --scheduler MultiStepLR --decay-milestones 150000 200000 \
  --lr 1e-2 \
  --optimizer NovoGrad \
  --train-data-path youtube/cut/cut_train.json echomsk6000/cut2/cut2_train.json echomsk6000/cut2/cut2_val.json echomsk6000/cut2/cut2_test.json  \
  --val-data-path data/clean_val.csv.json youtube/cut/cut_microval_10h.json echomsk6000/cut2/cut2_microval.json data/mixed_val.csv.json kontur_calls_micro/kontur_calls_micro.csv.json kontur_calls_micro/kontur_calls_micro.0.csv.json kontur_calls_micro/kontur_calls_micro.1.csv.json data/kfold_splits/22052020/valset_kfold_22052020.0_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020.1_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020_fold_0.csv.json \
  --val-iteration-interval 5000 \
  --fp16 O2 \
  --experiment-name clean_base_from_scratch \
  --window-size 0.02 \
  --iterations-per-epoch 5000 \
  --skip-on-epoch-end-evaluation \
  --epochs 1000 --exphtml= #\