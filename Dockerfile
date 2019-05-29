FROM nvidia/cuda:10.2-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update  -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    unrar \
    unzip \
    libnvinfer6=6.0.1-1+cuda10.2 \
    libnvinfer-dev=6.0.1-1+cuda10.2 \
    libnvinfer-plugin6=6.0.1-1+cuda10.2 \
    git && \
    apt-get clean -y

# Python
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b  && \
    rm -rf Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install -c anaconda -y \
      python=3.7.2 \
      pip

# JupyterLab
RUN conda install -c conda-forge jupyterlab

# Main frameworks
RUN pip install torch

# Install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
