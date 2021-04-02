set -e

CUDA_VISIBLE_DEVICES=0 python3 benchmark_repro.py \
  --onnx conv_fp16.onnx \
  --iterations 30 \
  --iterations-warmup 30 \
  --run-with-io-binding \
  -B 32 \
  -T 1664