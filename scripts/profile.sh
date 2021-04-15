set -e

TRACEFILE=data/profile.sqlite
TRACELOG=data/profile.txt
TRACEPYPROFLOG=data/profile.pyprof.txt
LOGDEST=cudnn_log.txt
CUBLAS_LOGDEST=cublas_log.txt
CUDNN_LOGDEST_DBG=$LOGDEST CUBLAS_LOGDEST_DBG=$CUBLAS_LOGDEST nvprof -s -f -o $TRACEFILE --profile-from-start off -- python3 benchmark.py --profile-cuda $@ &> $TRACELOG

# https://github.com/ezyang/nvprof2json
# then open it with chrome://tracing
python3 scripts/nvprof2json.py $TRACEFILE > $TRACEFILE.json

# pip install git+https://github.com/NVIDIA/PyProf.git

#python3 -m pyprof.parse $TRACEFILE > $TRACEFILE.dict
#python3 -m pyprof.prof -w 170 -c kernel,op,sil,tc,flops,bytes,device,stream,block,grid $TRACEFILE.dict > $TRACEPYPROFLOG


echo nvprof sqlite dump in $TRACEFILE, text trace in $TRACELOG
echo pyprof dict dump in $TRACEFILE.dict, text trace in $TRACEPYPROFLOG
echo open nvvp visual profiler as:
echo nvvp $TRACEFILE

#From https://github.com/NVIDIA/apex/tree/master/apex/pyprof
#You can choose which columns you'd like to display. Here's a list from calling python -m apex.pyprof.prof -h:
#idx:      Index
#seq:      PyTorch Sequence Id
#altseq:   PyTorch Alternate Sequence Id
#tid:      Thread Id
#layer:    User annotated NVTX string (can be nested)
#trace:    Function Call Trace
#dir:      Direction
#sub:      Sub Sequence Id
#mod:      Module
#op:       Operation
#kernel:   Kernel Name
#params:   Parameters
#sil:      Silicon Time (in ns)
#tc:       Tensor Core Usage
#device:   GPU Device Id
#stream:   Stream Id
#grid:     Grid Dimensions
#block:    Block Dimensions
#flops:    Floating point ops (FMA = 2 FLOPs)
#bytes:    Number of bytes in and out of DRAM
