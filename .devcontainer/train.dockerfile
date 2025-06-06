FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
LABEL maintainer="robot-squad"
LABEL description="Docker image for training vla models based on AigBot framework"

# Install dependencies
# python evdev, opengl, git
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-evdev \
    libgl1 \ 
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# copy requirement file to current workspace and install required python packages
COPY requirements.txt /tmp
RUN  pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# install flash attention
RUN pip install packaging ninja \
    && ninja --version; echo $?  # Verify Ninja --> should return exit code "0" \
    && pip install "flash-attn==2.5.5" --no-build-isolation

# install imgaug using conda
RUN conda config --add channels conda-forge \
    && conda install imgaug \
    && conda clean --all -y
    
# activate conda env 