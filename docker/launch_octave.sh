
# docker run -it --rm \
#     --shm-size=8G \
#     -e DISPLAY=$DISPLAY \
#     -v $(pwd):/workspace \
#     -w /workspace \
#     --name expr-GLR \
#     mtmiller/octave:latest

docker run -it --rm \
    --shm-size=1G \
    -e DISPLAY=$DISPLAY \
    -v $(pwd):/workspace \
    -w /workspace \
    --name expr-GLR \
    gnuoctave/octave:6.3.0 bash
