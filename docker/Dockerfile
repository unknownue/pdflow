
# https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.11-py3

LABEL author="unknownue <unknownue@outlook.com>" version="1.0"
 
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

ARG USER_ID=1000
ARG GROUP_ID=1001
ARG DOCKER_USER=unknownue
ARG DOCKER_PASSWORD=password

ENV PATH="/home/$DOCKER_USER/.local/bin:${PATH}"
ADD mirror-ubuntu2004.txt /etc/apt/sources.list


RUN apt update && \
    apt install -y --no-install-recommends sudo && \
    apt install -y --no-install-recommends neovim libcgal-dev && \
    # apt install -y xorg-dev libglu1 && \
    rm -rf /var/lib/apt/lists/*


# Docker user -------------------------------------------------------------------
# See also http://gbraad.nl/blog/non-root/user/inside-a-docker-container.html
RUN adduser --disabled-password --gecos '' $DOCKER_USER && \
    adduser $DOCKER_USER sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    echo "[core]editor=nvim" >> ~/.gitconfig && \
    git config --global user.email unknownue@outlook.com && \
    git config --global user.name unknownue
USER $DOCKER_USER

# Packages -----------------------------------------------------------------
RUN pip install --no-cache-dir --user pqi && pqi use aliyun
RUN pip install --no-cache-dir --user pytorch_lightning==1.5.3 scikit-learn rich && \
    pip install --user git+git://github.com/fwilliams/point-cloud-utils && \
    # https://github.com/unlimblue/KNN_CUDA
    pip install --no-cache-dir --user --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# For evaluation
RUN pip install --user "git+https://github.com/facebookresearch/pytorch3d.git@stable" && \
    git clone --recursive https://github.com/NVIDIAGameWorks/kaolin && cd kaolin && \
    git checkout v0.9.1 && IGNORE_TORCH_VER=1 python setup.py develop --user && \
    cd .. && git clone https://github.com/rusty1s/pytorch_cluster && \
    cd pytorch_cluster && python setup.py install --user

# For render figures
# RUN conda create -n blender_render python=3.7 && \
#     wget "https://download.blender.org/release/Blender2.79/blender-2.79b-linux-glibc219-x86_64.tar.bz2" && \
#     tar jxvf blender-2.79b-linux-glibc219-x86_64.tar.bz2 && \
#     wegt https://github.com/TylerGubala/blenderpy/releases/download/v2.91a0/bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl && \
#     pip install bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl && bpy_post_install
# RUN sudo apt install libglu1 libxi-dev libxrender1
# conda activate blender_render


# CMD ["bash"]
RUN python -c "import torch; print(torch.__config__.show())" && \
    python -m torch.utils.collect_env
