CUDA_VISIBLE_DEVICES=3 python3 benchmark.py \
  --checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr2e-5_wd1e-3_bs512_slow_fine_tune_after_uncertainty_self_train_checkpoint_epoch268_iter0370000.pt \
  --fp16 O2 \
  --frontend \
  --iterations 4 \
  --iterations-warmup 2 \
  -B 32 \
  -T 240

