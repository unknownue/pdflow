
docker run -id --rm \
   -e DISPLAY=unix$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v /home/unknownue/Workspace/Research:/workspace \
   --device /dev/nvidia0 \
   --device /dev/nvidia-uvm \
   --device /dev/nvidia-uvm-tools \
   --device /dev/nvidiactl \
   --gpus all \
   --name expr-totaldenoise \
   -e NVIDIA_DRIVER_CAPABILITIES=graphics,display,compute,utility \
   -w /workspace \
   --shm-size 8G \
   nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

   # tensorflow/tensorflow:1.13.1-devel-gpu-py3
# tensorflow/tensorflow:1.12.0-devel-gpu-py3