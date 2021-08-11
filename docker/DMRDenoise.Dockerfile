
FROM nvcr.io/nvidia/pytorch:21.05-py3

LABEL author="unknownue <unknownue@outlook.com>" version="1.0"
 
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

RUN apt update && \
    apt install -y ca-certificates sudo wget && \
    apt autoremove && apt clean && \
    rm -r /var/lib/apt/lists/*

RUN pip install --no-cache-dir --user scikit-learn==0.23.1 h5py==2.10.0 pytorch-lightning==0.7.6

RUN python -m torch.utils.collect_env
