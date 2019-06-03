FROM nvidia/cuda:10.0-cudnn7-devel

# Install system packages
RUN apt-get update  -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    unrar \
    git && \
    apt-get clean -y

# Install Python
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update  -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip && \
    apt-get clean -y

RUN pip3 install --upgrade pip setuptools wheel

# Install requirements
COPY requirements.txt .
COPY requirements-gpu.txt .
RUN pip install -r requirements-gpu.txt
RUN pip install -r requirements.txt
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Enable GPU access from ssh login into the Docker container
RUN echo "ldconfig" >> /etc/profile
