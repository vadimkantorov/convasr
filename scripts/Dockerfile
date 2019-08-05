FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt update -y && apt install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  curl \
  ca-certificates \
  apt-utils \
  bzip2 \
  tree \
  htop \
  bmon \
  iotop \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  cmake \
  sox \
  libsox-dev \
  libsox-fmt-all \
  vim \
  sudo \
  wget \
  gdb \
  && apt clean && rm -rf /var/lib/apt/lists/*

RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh
ENV PATH=/miniconda/bin:$PATH

RUN /miniconda/bin/conda install -y ipython matplotlib scipy scikit-learn numba

# Prerequisites for building PyTorch from source (just in case)
RUN /miniconda/bin/conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
RUN conda install  -y -c pytorch magma-cuda100 

RUN conda install -y -c pytorch cudatoolkit=10.0 pytorch-nightly=1.2.0.dev20190805

RUN pip install cython librosa python-levenshtein tqdm
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex
