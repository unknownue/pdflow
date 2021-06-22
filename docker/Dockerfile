
# https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.05-py3

LABEL author="unknownue <unknownue@outlook.com>" version="1.0"
 
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

ARG USER_ID=1000
ARG GROUP_ID=1001
ARG DOCKER_USER=unknownue
ARG DOCKER_PASSWORD=password

ENV PATH="/home/$DOCKER_USER/.local/bin:${PATH}"
# ADD mirror-ubuntu2004.txt /etc/apt/sources.list


RUN apt update && \
    apt install -y --no-install-recommends sudo && \
    apt install -y --no-install-recommends nvim && \
    rm -rf /var/lib/apt/lists/*


# Docker user -------------------------------------------------------------------
# See also http://gbraad.nl/blog/non-root/user/inside-a-docker-container.html
RUN adduser --disabled-password --gecos '' $DOCKER_USER && \
    adduser $DOCKER_USER sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    echo "[core]editor=nvim" >> ~/.gitconfig
USER $DOCKER_USER

# Packages -----------------------------------------------------------------
RUN pip install --no-cache-dir --user pqi && pqi use aliyun
RUN pip install --no-cache-dir --user pytorch_lightning==1.3.0


# CMD ["bash"]
RUN python -c "import torch; print(torch.__config__.show())" && \
    python -m torch.utils.collect_env
