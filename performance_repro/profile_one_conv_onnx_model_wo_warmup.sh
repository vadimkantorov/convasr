set -e

TRACEFILE=profile_one_conv_onnx_wo_warmup.sqlite
TRACELOG=profile_one_conv_onnx_wo_warmup.txt
TRACEPYPROFLOG=profile_one_conv_onnx_wo_warmup.pyprof.txt
CUDNN_LOGDEST=profile_one_conv_onnx_wo_warmup_cudnn_dbg.txt
CUBLAS_LOGDEST=profile_one_conv_onnx_wo_warmup_cublas_dbg.txt

CUDA_VISIBLE_DEVICES=0 CUDNN_LOGINFO_DBG=1 CUDNN_LOGDEST_DBG=$CUDNN_LOGDEST CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=$CUBLAS_LOGDEST nvprof -f -o $TRACEFILE -s --devices 0 --profile-from-start off -- python3 benchmark_repro.py \
  --onnx conv_fp16.onnx \
  --iterations 1 \
  --iterations-warmup 0 \
  --profile-cuda \
  --channels 896 \
  --run-with-io-binding \
  -B 32 \
  -T 832 &> $TRACELOG

python3 nvprof2json.py $TRACEFILE > $TRACEFILE.json
