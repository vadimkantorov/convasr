set -e

CUDA_VISIBLE_DEVICES=0 python3 benchmark_repro.py \
  --iterations 30 \
  --channels 896 \
  --iterations-warmup 30 \
  -B 32 \
  -T 832