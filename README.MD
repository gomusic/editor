# Video Processing with Matting and Chroma Key Replacement

This project allows you to process videos using a matting technique (from the Robust Video Matting model) and apply chroma key replacement to specific regions of the video. The chroma key effect supports dynamic zooming during the video playback, controlled via parameters.

## Table of Contents
[Features](#Features) \
[Installation](#Installation)\
[Minimum Requirements](#Minimum-Requirements)\
[Usage](#Usage)\
[Arguments](#Arguments)\
[Example](#Example)\
[Replace Color](#Replace-color)\
[Notes](#Notes)

## Features

- [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting): Automatically segments and extracts objects from videos (such as people) based on the selected matting model.
- Chroma Key Replacement: Replaces the green/blue screen regions of a video with a background video or image.
- Dynamic Zooming: The script allows dynamic zoom effects during playback, with customizable zoom settings.
- Automatic Mask Creation: Converts videos into a sequence of PNG images with alpha transparency for further editing or compositing.

## Installation

- Install dependencies:
    ```bash
        pip install -r requirements.txt
    ```

- Download the model weights for the Robust Video Matting model:
    You can download [rvm_resnet50.pth or rvm_mobilenetv3.pth](https://github.com/PeterL1n/RobustVideoMatting?tab=readme-ov-file#download) from the [official Robust Video Matting repository](https://github.com/PeterL1n/RobustVideoMatting).
    Place the downloaded .pth model in the root directory of the project, or specify its path in the command line arguments.

- Install [pytorch](https://pytorch.org/) (Recommendation: 2.4.1+cu118)

## Minimum Requirements

- Python: 3.6 or higher.
- CUDA: 10.1 or higher (if using a GPU).
- PyTorch: 1.7 or later (Recommendation: 2.4.1+cu118).
- FFmpeg: Must be installed for handling video inputs and outputs.
- GPU: An NVIDIA GPU is recommended for optimal performance, although CPU usage is possible.

Recommendation: **PyTorch:** 2.4.1, **CUDA:** 11.8.

## Usage

### To run the script, use the following command format:

```bash

python pipeline.py --original_video <path_to_original_video> \
                   --full_background <path_to_full_background_video> \
                   --phone_background <path_to_phone_background_video> \
                   --model_path <path_to_model> \
                   --zoom_scale <initial_zoom_scale> \
                   --zoom_increment <zoom_increment> \
                   --lower_green 35 40 40 \
                   --upper_green 85 255 255 
```
## Arguments
```text
    --original_video: (Required) Path to the original video file that you want to process.
    --full_background: (Required) Path to the full background video that will replace the chroma-keyed regions of the original video.
    --model_path: (Optional) Path to the Robust Video Matting model file. rvm_resnet50.pth or rvm_mobilenetv3.pth. Default: rvm_resnet50.pth.
    --zoom_scale: (Optional) The initial zoom scale for the chroma key effect. Default: 1.0.
    --zoom_increment: (Optional) The zoom increment applied per frame for the dynamic zoom effect. Default: 0.2.
    --output_video_name (Optional) Title of the final video without extension (default: output_video).
    --lower_blue: (Optional) Lower HSV range for the blue color used in chroma key processing, provided as three integers. 
                  Default: [80, 50, 80].
    --upper_blue: (Optional) Upper HSV range for the blue color used in chroma key processing, provided as three integers. 
                  Default: [130, 255, 255].
    --lower_green: (Optional) Lower HSV range for the green color used in chroma key processing, provided as three integers. 
                   Default: [35, 40, 40].
    --upper_green: (Optional) Upper HSV range for the green color used in chroma key processing, provided as three integers. 
                   Default: [85, 255, 255].
```
    
### Description of Chroma Keying and Zoom Parameters:
```text
    Zoom Scale: This controls the initial zoom level on the phone screen’s background. A value of 1.0 means no zoom, while higher values will zoom in.
    Zoom Increment: This controls how fast the zoom increases during the video. A higher increment will result in faster zooming.
```

## Example

### To process a video called input.mp4, apply a background from full_bg.mp4:

```bash

python pipeline.py --original_video input.mp4 \
                   --full_background full_bg.mp4 \
                   --phone_background phone_bg.mp4 \
                   --model_path rvm_resnet50.pth 
```
### Example workflow:

***Step 1:*** For each video that is being processed for the first time, the [replace_color](#Replace-color) script is run first.\
***Step 2:*** Next, the script removes the chroma key color (green or blue) from the original video and replaces it with transparent layers.\
***Step 3:*** Using the selected background video, the chromakey effect is applied to the transparent areas.\
***Step 4:*** Zoom is dynamically applied to the phone screen, creating a zoom effect during video playback.

#### Before:
![](README_ASSETS/before.gif)

#### After:
![](README_ASSETS/after.gif)

## Replace color

This script finds dots of a different color on the background and paints them in the background color. \
The result will be a video with the replaced color in the background and named output_name_replace_color.mp4

### Before replace
![](README_ASSETS/before.gif)

### After replace
![](README_ASSETS/after_replace.gif)

## Notes

The sequence/ folder will contain PNG files (with alpha channels) of each processed frame after applying the matting model.\
The finished video will have the default name output_video.mp4 and will be saved in the current directory.\
The default matting model is rvm_resnet50.pth, but you can also use the smaller rvm_mobilenetv3.pth model for faster processing if needed.\

Feel free to adjust the zoom parameters as per your creative requirements to achieve the desired effect.
