for b in 1 4 8 16 24 32 48 64 96 128 192 256 512 1024; do \
for t in 10.23 30.07 60.15 119.99 239.99 359.99 479.99 959.99 1919.99 3839.99; do \
CUDA_VISIBLE_DEVICES=0 python3 benchmark.py --fp16 O2  --model JasperNetBig  --iterations 9   --iterations-warmup 3  --frontend  -B $b  -T $t --output-path benchmark_grid_search_pytorch.csv
done; \
done