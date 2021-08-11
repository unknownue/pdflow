
From nvcr.io/nvidia/tensorflow:20.12-tf1-py3

LABEL author="unknownue <unknownue@outlook.com>" version="1.0"
 
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# RUN apt update && \
#     apt install -y --no-install-recommends sudo && \
#     rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir point_cloud_utils
