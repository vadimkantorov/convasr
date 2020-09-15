python3 train.py $@ \
  --githttp https://github.com/vadimkantorov/convasr/commit/%h \
  --verbose --lang ru \
  --model JasperNetBig \
  --train-batch-size 1536 --val-batch-size 512 \
  --iterations 700000 \
  --lr 1e-4 \
  --optimizer NovoGrad \
  --train-data-path youtube/cut/cut_train.json youtube_lowpass/cut/cut_train.json echomsk6000/cut2/cut2_train.json echomsk6000/cut2/cut2_val.json echomsk6000/cut2/cut2_test.json  \
  --val-data-path data/clean_val.csv.json data/mixed_val.csv.json kontur_calls_micro/kontur_calls_micro.csv.json kontur_calls_micro/kontur_calls_micro.0.csv.json kontur_calls_micro/kontur_calls_micro.1.csv.json youtube/cut/cut_microval.json data/kfold_splits/22052020/valset_kfold_22052020.0_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020.1_fold_0.csv.json data/kfold_splits/22052020/valset_kfold_22052020_fold_0.csv.json \
  --val-iteration-interval 5000  \
  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-3_wd1e-3_bs1536____summary_youtube_ws_additional_radio_lr001/checkpoint_epoch307_iter0625000.pt \
  --fp16 O2 \
  --experiment-name summary_youtube_ws_additional_radio_lr001 \
  --window-size 0.04 \
  --iterations-per-epoch 5000 \
  --skip-on-epoch-end-evaluation \
  --epochs 1000 --exphtml= #\

# data/experiments/JasperNetBig_NovoGrad_lr1e-3_wd1e-3_bs1536____summary_youtube_ws_additional_radio_lr001/checkpoint_epoch307_iter0625000.pt
# data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs1536____summary_youtube_ws_additional_radio_lr001/checkpoint_epoch294_iter0560000.pt
# data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs1536____summary_youtube_codecs_ws_radio/checkpoint_epoch269_iter0435000.pt

# init pt data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs1536____summary/checkpoint_epoch262_iter0400000.pt
# data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs1536____summary_youtube_codecs_ws_radio/checkpoint_epoch269_iter0435000.pt
# data/experiments/JasperNetBig_NovoGrad_lr1e-3_wd1e-3_bs1536____summary_youtube_ws_radio/checkpoint_epoch282_iter0500000.pt

#--scheduler MultiStepLR --decay-milestones 300000 400000 \
#echomsk6000/cut2/cut2_train.json
