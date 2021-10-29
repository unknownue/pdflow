
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

LABEL author="unknownue <unknownue@outlook.com>" version="1.0"
 
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

ARG USER_ID=1000
ARG GROUP_ID=1001
ARG DOCKER_USER=unknownue
ARG DOCKER_PASSWORD=password

ENV PATH="/home/$DOCKER_USER/.local/bin:${PATH}"

ADD mirror-ubuntu1604.txt /etc/apt/sources.list

RUN apt update && \
    apt install -y --no-install-recommends sudo wget unzip git

RUN apt install -y --no-install-recommends python2.7 && \
    wget https://bootstrap.pypa.io/pip/2.7/get-pip.py && \
    python2.7 get-pip.py && rm get-pip.py && \
    ln -s /usr/bin/python2.7 /usr/bin/python && \
    pip install --no-cache-dir --user numpy==1.16.4

# Docker user -------------------------------------------------------------------
# See also http://gbraad.nl/blog/non-root/user/inside-a-docker-container.html
RUN adduser --disabled-password --gecos '' $DOCKER_USER && \
    adduser $DOCKER_USER sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    echo "[core]editor=nvim" >> ~/.gitconfig && \
    git config --global user.email unknownue@outlook.com && \
    git config --global user.name unknownue
USER $DOCKER_USER

RUN python2.7 -m pip install --no-cache-dir tensorflow_gpu==1.12.0











# RUN wget https://github.com/bazelbuild/bazel/releases/download/0.15.0/bazel-0.15.0-installer-linux-x86_64.sh && \
#     bash bazel-0.15.0-installer-linux-x86_64.sh && rm bazel-0.15.0-installer-linux-x86_64.sh

# RUN ln -s /usr/local/cuda/lib64/libcudart.so.11.1.74 /usr/local/cuda/lib64/libcudart.so.11.1 && \
#     ln -s /usr/local/cuda/lib64/libcublas.so.11 /usr/local/cuda/lib64/libcublas.so.11.1 && \
#     ln -s /usr/local/cuda/lib64/libcusolver.so.11 /usr/local/cuda/lib64/libcusolver.so.11.1 && \
#     ln -s /usr/local/cuda/lib64/libcurand.so.10 /usr/local/cuda/lib64/libcurand.so.11.1 && \
#     ln -s /usr/local/cuda/lib64/libcufft.so.10 /usr/local/cuda/lib64/libcufft.so.11.1

# RUN wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v1.12.0.zip && \
#     unzip v1.12.0.zip && rm v1.12.0.zip && \
#     cd tensorflow-1.12.0 && \
#     ./configure && \   # cuda 11.1, cudnn8, capability 8.0,8.6+PTX, default nvcc, no MPI

# edit "cuda_version = _cuda_version(repository_ctx, cuda_toolkit_path, cpu_value)" to "cuda_version = '11.1'" in tensorflow-1.12.0/third_party/gpus/cuda_configure.bzl
# edit "cudnn_version = _cudnn_version(repository_ctx, cudnn_install_basedir, cpu_value)" to "cudnn_version = '8.0.5'"
# RUN bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
# Update checksum in third_party/icu/workspace.bzl

