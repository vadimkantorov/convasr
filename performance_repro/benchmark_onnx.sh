set -e

CUDA_VISIBLE_DEVICES=0 python3 benchmark_repro.py \
  --onnx conv_fp16.onnx \
  --iterations 10 \
  --iterations-warmup 10 \
  -B 32 \
  -T 1664