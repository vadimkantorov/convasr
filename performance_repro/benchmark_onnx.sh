set -e

python3 benchmark_repro.py \
  --fp16 O2 \
  --model OneConvModel \
  --onnx conv_fp16.onnx \
  --iterations 10 \
  --iterations-warmup 10 \
  -B 32 \
  -T 16