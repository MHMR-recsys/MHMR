FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update

# ==================================================================
# tools
# ------------------------------------------------------------------

RUN    APT_INSTALL="apt-get install -y --no-install-recommends" && \
DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    apt-utils \
    build-essential \
    ca-certificates \
    cmake \
    wget \
    git \
    vim \
	ffmpeg \
	libopenmpi-dev \
	tmux \
	htop \
	libosmesa6-dev \
	libgl1-mesa-glx \
	libglfw3 \
    imagemagick \
    libopencv-dev \
    python-opencv \
    curl \
    libjpeg-dev \
    libpng-dev \
    axel \
    zip \
    unzip \
    && pip install tensorboard \
	&& pip install tensorboardX \
    && pip install tqdm \
    && pip install opencv-python \
    && pip install pandas \
    && pip install scikit-learn \
    && pip install scipy \
    && pip install matplotlib \
    && pip install seaborn \
    && pip install matplotlib \
    && pip install hydra-core


# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

