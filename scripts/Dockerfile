FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

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
  openssh-client \
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

RUN pip install ipython matplotlib seaborn

# Prerequisites for building PyTorch from source (just in case)
RUN /miniconda/bin/conda install -y scipy scikit-learn numba numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
RUN conda install -y -c pytorch magma-cuda101 

# RUN conda install -y pytorch cudatoolkit=10.1 -c pytorch
RUN conda install -y pytorch cudatoolkit=10.1 -c pytorch-nightly

# RUN pip install torchaudio -f https://download.pytorch.org/whl/torch_stable.html --no-dependencies
RUN pip install git+https://github.com/pytorch/audio

RUN pip install future cython seaborn librosa tqdm wget tensorboard python-levenshtein 
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex
RUN pip install git+https://github.com/parlance/ctcdecode
RUN pip install awscli

RUN pip install sentencepiece

RUN  echo "    IdentityFile /root/.ssh/id_rsa" >> /etc/ssh/ssh_config
