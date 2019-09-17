FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
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
  vim unzip \
  sudo \
  wget \
  gdb parallel \
  libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh
ENV PATH=/miniconda/bin:$PATH

RUN /miniconda/bin/conda install -y ipython matplotlib scipy scikit-learn numba

# Prerequisites for building PyTorch from source (just in case)
RUN /miniconda/bin/conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
RUN conda install -y -c pytorch magma-cuda100 

RUN conda install -y pytorch cudatoolkit=10.0 -c pytorch
RUN pip install torchaudio -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install future cython librosa python-levenshtein tqdm wget tensorboard==1.14.0
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex
RUN pip install git+https://github.com/parlance/ctcdecode
RUN pip install awscli

RUN pip install sentencepiece
