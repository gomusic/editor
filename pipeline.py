from RobustVideoMatting.inference import  convert_video
from RobustVideoMatting.model import MattingNetwork
import torch


model = MattingNetwork('resnet50').eval().cpu()  # or "resnet50"
model.load_state_dict(torch.load('rvm_resnet50.pth'))



"""
convert_video(
    model,                           # The model, can be on any device (cpu or cuda).
    #input_source='./../BackgroundMattingV2/test/headphones.mp4',        # A video file or an image sequence directory.
    input_source='./../output.mp4',        # A video file or an image sequence directory.
    output_type='video',             # Choose "video" or "png_sequence"
    output_composition='com.mov',    # File path if video; directory path if png sequence.
    output_alpha="pha.mov",          # [Optional] Output the raw alpha prediction.
    output_foreground="output_foreground.mov",          # [Optional] Output the raw alpha prediction.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
    seq_chunk=24,                    # Process n frames at once for better parallelism.
)
"""
convert_video(
    model,                           # The model, can be on any device (cpu or cuda).
    #input_source='./../BackgroundMattingV2/test/headphones.mp4',        # A video file or an image sequence directory.
    input_source='./../output.mp4',        # A video file or an image sequence directory.
    output_type='png_sequence',             # Choose "video" or "png_sequence"
    output_composition='./../sequence',    # File path if video; directory path if png sequence.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
    seq_chunk=24,                    # Process n frames at once for better parallelism.
)