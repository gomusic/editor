`ffmpeg -loop 1 -i bg.png -c:v libx264 -t 3 -pix_fmt yuv420p -vf "scale=3840:2160" output_video.mp4`


1. For yolov5 + torch

```
pip install opencv-python opencv-python-headless
pip install torch torchvision
For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
For GPU support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

pip install yolov5
pip install huggingface-hub
pip install numpy
```