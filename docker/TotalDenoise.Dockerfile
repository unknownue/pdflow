
From nvidia/cuda:11.2.0-cudnn8-devel-ubuntu16.04

LABEL author="unknownue <unknownue@outlook.com>" version="1.0"
 
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

RUN apt update && \
    apt install -y --no-install-recommends sudo wget git p7zip-full unzip build-essential && \
    apt install python3 python3-pip python3-numpy && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/

RUN wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v1.13.1.zip && \
    7z x v1.13.1.zip && rm v1.13.1.zip && \
    wget https://github.com/bazelbuild/bazel/releases/download/0.20.0/bazel-0.20.0-installer-linux-x86_64.sh && \
    bash bazel-0.20.0-installer-linux-x86_64.sh && \
    source /usr/local/lib/bazel/bin/bazel-complete.bash && \
    rm bazel-0.20.0-installer-linux-x86_64.sh && \
    mkdir /opt/tensorflow-1.13.1/build

WORKDIR /opt/tensorflow-1.13.1/build

# Enable CUDA and cudnn 8 support; Specify 8.6 compute capabilities; no tensorRT, no nccl, no xla, no opencl, no ROCm
RUN ../configure && \
    ln -s /usr/local/cuda/lib64/libcudart.so.11.2.72 /usr/local/cuda/lib64/libcudart.so.11.2 && \
    ln -s /usr/local/cuda/lib64/libcublas.so.11 /usr/local/cuda/lib64/libcublas.so.11.2 && \
    ln -s /usr/local/cuda/lib64/libcusolver.so.11 /usr/local/cuda/lib64/libcusolver.so.11.2 && \
    ln -s /usr/local/cuda/lib64/libcurand.so.11 /usr/local/cuda/lib64/libcurand.so.11.2 && \
    ln -s /usr/local/cuda/lib64/libcurand.so.11 /usr/local/cuda/lib64/libcurand.so.11.2 && \
    ln -s /usr/local/cuda/lib64/libcufft.so.10 /usr/local/cuda/lib64/libcufft.so.11.2 && \
    # change checksum in /.cache/bazel/_bazel_root/???/external/org_tensorflow/third_party/icu/workspace.bzl
    bazel build --config=opt --config=cuda --config=noaws --config=nonccl --config=noignite //tensorflow/tools/pip_package:build_pip_package
