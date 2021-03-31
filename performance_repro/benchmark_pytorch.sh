set -e

python3 benchmark_repro.py \
  --fp16 O2 \
  --model OneConvModel \
  --iterations 1 \
  --iterations-warmup 4 \
  -B 32 \
  -T 16