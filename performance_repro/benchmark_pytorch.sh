set -e

CUDA_VISIBLE_DEVICES=0 python3 benchmark_repro.py \
  --fp16 O2 \
  --model OneConvModel \
  --iterations 10 \
  --iterations-warmup 10 \
  -B 32 \
  -T 16