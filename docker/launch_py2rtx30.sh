
docker run -id --rm \
   -e DISPLAY=unix$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v /home/unknownue/Workspace/Research:/workspace \
   --device /dev/nvidia0 \
   --device /dev/nvidia-uvm \
   --device /dev/nvidia-uvm-tools \
   --device /dev/nvidiactl \
   --gpus all \
   --name expr-py2rtx30 \
   -e NVIDIA_DRIVER_CAPABILITIES=graphics,display,compute,utility \
   -w /workspace \
   --shm-size 8G \
   nvcr.io/nvidia/tensorflow:20.01-tf1-py2
