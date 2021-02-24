set -e

TRACEFILE=profile_one_layer_pytorch.sqlite
TRACELOG=profile_one_layer_pytorch.txt
TRACEPYPROFLOG=profile_one_layer_pytorch.pyprof.txt
CUDNN_LOGDEST=profile_one_layer_pytorch_cudnn_dbg.txt
CUBLAS_LOGDEST=profile_one_layer_pytorch_cublas_dbg.txt

CUDA_VISIBLE_DEVICES=0 CUDNN_LOGINFO_DBG=1 CUDNN_LOGDEST_DBG=$CUDNN_LOGDEST CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=$CUBLAS_LOGDEST nvprof -f -o $TRACEFILE -s --devices 0 --profile-from-start off -- python3 benchmark.py \
  --fp16 O2 \
  --model ProfilableModel \
  --iterations 1 \
  --iterations-warmup 4 \
  --profile-cuda \
  -B 32 \
  -T 16 &> $TRACELOG

python3 scripts/nvprof2json.py $TRACEFILE > $TRACEFILE.json
