python3 benchmark.py \
  --model JasperNetBig \
  --fp16 O2 \
  --frontend \
  --iterations 10 \
  --iterations-warmup 4 \
  --profile-cuda \
  -B 32 \
  -T 160
# RTF 4950

python3 benchmark.py \
  --model JasperNetBig \
  --fp16 O2 \
  --frontend \
  --iterations 10 \
  --iterations-warmup 4 \
  --profile-cuda \
  -B 32 \
  -T 160 \
  --satisfy-features-divisibility-by-32
# RTF 5177