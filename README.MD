# Video Processing with Matting and Chroma Key Replacement

This project allows you to process videos using a matting technique (from the Robust Video Matting model) and apply chroma key replacement to specific regions of the video. The chroma key effect supports dynamic zooming during the video playback, controlled via parameters.

## Table of Contents
[Features](#Features) \
[Installation](#Installation)\
[Usage](#Usage)\
[Arguments](#Arguments)\
[Example](#Example)\
[Notes](#)

## Features

    Robust Video Matting: Automatically segments and extracts objects from videos (such as people) based on the selected matting model.
    Chroma Key Replacement: Replaces the green/blue screen regions of a video with a background video or image.
    Dynamic Zooming: The script allows dynamic zoom effects during playback, with customizable zoom settings.
    Automatic Mask Creation: Converts videos into a sequence of PNG images with alpha transparency for further editing or compositing.

## Installation

- Install dependencies:
    ```bash
        pip install -r requirements.txt
    ```

- Download the model weights for the Robust Video Matting model:
    You can download [rvm_resnet50.pth or rvm_mobilenetv3.pth](https://github.com/PeterL1n/RobustVideoMatting?tab=readme-ov-file#download) from the [official Robust Video Matting repository](https://github.com/PeterL1n/RobustVideoMatting).
    Place the downloaded .pth model in the root directory of the project, or specify its path in the command line arguments.

- Install [pytorch](https://pytorch.org/) (Recommendation: 2.4.1+cu118)

## Usage

### To run the script, use the following command format:

```bash

python pipeline.py --original_video <path_to_original_video> \
                   --full_background <path_to_full_background_video> \
                   --phone_background <path_to_phone_background_video> \
                   --model_path <path_to_model> \
                   --zoom_scale <initial_zoom_scale> \
                   --zoom_increment <zoom_increment>
```
## Arguments
```text
    --original_video: (Required) Path to the original video file that you want to process.
    --full_background: (Required) Path to the full background video that will replace the chroma-keyed regions of the original video.
    --model_path: (Optional) Path to the Robust Video Matting model file. rvm_resnet50.pth or rvm_mobilenetv3.pth. Default: rvm_resnet50.pth.
    --zoom_scale: (Optional) The initial zoom scale for the chroma key effect. Default: 1.0.
    --zoom_increment: (Optional) The zoom increment applied per frame for the dynamic zoom effect. Default: 0.2.
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
### Example Workflow:

***Step 1:*** The script first removes the chroma key color (green or blue) from the original video and replaces it with transparent layers.\
***Step 2:*** Using the selected background video, the chroma key effect is applied to the transparent regions.\
***Step 3:*** The zoom is dynamically applied to the phone’s screen, creating a zoom-in effect throughout the video playback.

#### Before:
![](examples_gif/before.gif)

#### After:
![](examples_gif/after.gif)

## Notes

The sequence/ folder will contain PNG files (with alpha channels) of each processed frame after applying the matting model.\
The final video output will be named input_output_detected_video.mp4 and saved in the current directory.\
The default matting model is rvm_resnet50.pth, but you can also use the smaller rvm_mobilenetv3.pth model for faster processing if needed.\

Feel free to adjust the zoom parameters as per your creative requirements to achieve the desired effect.