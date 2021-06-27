
docker run -id --rm \
   -e DISPLAY=unix$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v $(pwd):/workspace \
   --device /dev/nvidia0 \
   --device /dev/nvidia-uvm \
   --device /dev/nvidia-uvm-tools \
   --device /dev/nvidiactl \
   --gpus all --name expr-pdflow \
   -e NVIDIA_DRIVER_CAPABILITIES=graphics,display,compute,utility \
   -w /workspace \
   --shm-size 8G \
   unknownue/pdflow
