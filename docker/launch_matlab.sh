docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro --shm-size=512M mathworks/matlab:latest
# docker run -it --rm \
#     --shm-size=512M \
#     -e DISPLAY=$DISPLAY \
#     -w /workspace \
#     --name expr-GLR \
#     mathworks/matlab:latest -shell

    # -v $(pwd):/workspace \

# docker run -it --rm \
#     --shm-size=512M \
#     --name expr-GLR \
#     -v $(pwd):/workspace \
#     -w /workspace \
#     demartis/matlab-runtime:latest

