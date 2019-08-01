FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt update -y && apt install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ libglib2.0-0 libsm6 libxext6 libxrender-dev cmake sox libsox-dev libsox-fmt-all vim sudo wget gdb

RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh
ENV PATH=/miniconda/bin:$PATH

RUN /miniconda/bin/conda install -y ipython matplotlib scipy scikit-learn numba

# Prerequisites for building PyTorch from source (just in case)
RUN /miniconda/bin/conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
RUN conda install pytorch -y -c magma-cuda100 

RUN conda install pytorch -y -c pytorch

RUN pip install cython librosa librosa python-levenshtein tqdm
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex
