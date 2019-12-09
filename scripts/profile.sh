if [ ! -z "$1" ]; then
	TRACEFILE="-o $1"
fi

#TODO: set up hw counters https://github.com/NVIDIA/apex/tree/master/apex/pyprof#hardware-counters

#TODO: use https://github.com/NVIDIA/apex/tree/master/apex/pyprof

nvprof --print-gpu-trace $TRACEFILE python benchmark.py
