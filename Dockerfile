#####
## Author: RAFAYAAMIR
## Date:   Sep 25 2024
#####

# BASE IMAGE HAVING CUDA 11.3.1 AND UBUNTU 20.04
FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 AS x86_64_build

ENV DEBIAN_FRONTEND=noninteractive
ENV BASE_DIR=/workspace
WORKDIR ${BASE_DIR}


RUN apt-get update --fix-missing && apt-get install -y \
    software-properties-common \
    build-essential \
    gnupg2 \
    wget \
    unzip \
    curl \
    python3-pip \
    git

COPY requirements.txt .
RUN pip3 --default-timeout=1000 install -r requirements.txt

RUN git clone https://github.com/gurkirt/3D-RetinaNet.git ${BASE_DIR}/3D-RetinaNet


ENV PYTHONPATH=/workspace
ENV PYTHONIOENCODING=UTF-8