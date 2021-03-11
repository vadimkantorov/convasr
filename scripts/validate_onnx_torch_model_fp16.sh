set -e

python3 train.py $@ \
  --device cuda \
  --frontend-in-model \
  --num-workers 32 \
  --fp16 O2 \
  --val-batch-size 2 \
  --checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs512____rerun_finetune_after_self_train_epoch183_iter0290000.pt \
  --val-data-path domain_set/transcripts/16082020/valset_16082020.0.csv.json