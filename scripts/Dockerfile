ARG CUDAVERSIONPOINT
FROM nvidia/cuda:$CUDAVERSIONPOINT-cudnn7-devel-ubuntu18.04
ARG CUDAVERSION
ARG CUDAVERSIONPOINT

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -y && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  curl \
  ca-certificates \
  apt-utils \
  bzip2 unzip p7zip-full \
  tree \
  htop \
  bmon \
  tmux \
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
  ffmpeg libavcodec-dev libavformat-dev libavfilter-dev \
  vim \
  sudo \
  wget \
  gdb strace parallel \
  libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN  echo "    IdentityFile /root/.ssh/id_rsa" >> /etc/ssh/ssh_config

RUN curl -L -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Prerequisites for building PyTorch from source (just in case)
RUN conda install -y scipy scikit-learn numba numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing pillow

RUN pip install ipython seaborn altair
RUN pip install matplotlib #https://github.com/matplotlib/matplotlib/archive/v3.2.0rc2.tar.gz

RUN conda install -y -c pytorch magma-cuda${CUDAVERSION}

RUN conda install -y pytorch cudatoolkit=$CUDAVERSIONPOINT -c pytorch
# RUN conda install -y pytorch cudatoolkit=$CUDAVERSIONOPOINT -c pytorch-nightly

# RUN pip install torchaudio -f https://download.pytorch.org/whl/torch_stable.html --no-dependencies
RUN pip install git+https://github.com/pytorch/audio --no-dependencies

RUN pip install future cython seaborn librosa tqdm wget tensorboard python-levenshtein 
RUN pip install awscli --ignore-installed PyYAML

RUN pip install sentencepiece
RUN pip install webrtcvad
RUN pip install onnxruntime
RUN pip install yapf

RUN git clone https://github.com/parlance/ctcdecode --recursive && sed -i 's/11/14/g' ctcdecode/build.py && pip install ./ctcdecode
RUN pip install cxxfilt && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex
