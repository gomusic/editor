`ffmpeg -loop 1 -i bg.png -c:v libx264 -t 3 -pix_fmt yuv420p -vf "scale=3840:2160" output_video.mp4`

### Using:
1. Install ```pip install -r requirements.txt```
2. For torch ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
3. Specify video paths in the pipeline file
4. Run the pipeline file