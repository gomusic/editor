import os
import torch
import argparse
from RobustVideoMatting.inference import convert_video
from RobustVideoMatting.model import MattingNetwork
from replace_color import replace
from chroma_key_replacer import chroma_replace
import numpy as np

# Define the color ranges in HSV
lower_blue = np.array([80, 50, 80])
upper_blue = np.array([130, 255, 255])
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Video Processing with Matting and Chroma Key Replacement")

    # Define named arguments
    parser.add_argument('--original_video', type=str, required=True, help="Path to the original video file")
    parser.add_argument('--full_background', type=str, required=True, help="Path to the full background video file")
    parser.add_argument('--phone_background', type=str, help="Path to the phone background video file")
    parser.add_argument('--model_path', type=str, default='rvm_resnet50.pth',
                        help="Path to the model file (rvm_resnet50.pth or rvm_mobilenetv3.pth)")
    parser.add_argument('--zoom_scale', type=float, default=1.0, help="Initial zoom scale for chroma key effect")
    parser.add_argument('--zoom_increment', type=float, default=0.2, help="Zoom increment during the effect")
    parser.add_argument('--output_video_name', type=str, default='output_video', help="Title of the final video without extension (default: output_video)")

    # Add command-line arguments for color ranges
    parser.add_argument('--lower_blue', type=int, nargs=3, default=[80, 50, 80],
                        help="Lower HSV range for blue (default: [80, 50, 80])")
    parser.add_argument('--upper_blue', type=int, nargs=3, default=[130, 255, 255],
                        help="Upper HSV range for blue (default: [130, 255, 255])")
    parser.add_argument('--lower_green', type=int, nargs=3, default=[35, 40, 40],
                        help="Lower HSV range for green (default: [35, 40, 40])")
    parser.add_argument('--upper_green', type=int, nargs=3, default=[85, 255, 255],
                        help="Upper HSV range for green (default: [85, 255, 255])")

    return parser.parse_args()

# Main function
def main():
    args = parse_arguments()

    # Extract filenames and paths from arguments
    original_video_path = args.original_video
    output_video = f'{os.path.splitext(os.path.basename(original_video_path))[0]}_output.mp4'
    output_composition = f'./sequence/{os.path.splitext(output_video)[0]}'
    full_background_path = args.full_background
    phone_background_path = args.phone_background

    # Convert color ranges from the arguments
    lower_blue = np.array(args.lower_blue)
    upper_blue = np.array(args.upper_blue)
    lower_green = np.array(args.lower_green)
    upper_green = np.array(args.upper_green)

    # Load the matting model
    model = MattingNetwork('resnet50').eval().cpu()
    model.load_state_dict(torch.load(args.model_path))

    # Check if output video exists, if not, replace color in the original video
    if not os.path.exists(output_video):
        replace(original_video_path, args.output_video_name, lower_green, upper_green)

    # Create output composition directory if it doesn't exist
    if not os.path.exists(output_composition):
        os.makedirs(output_composition)

        # Convert video using the matting model if composition is missing
        convert_video(
            model=model,  # Matting model
            input_source=output_video,  # Video file or image sequence
            output_type='png_sequence',  # Specify output as PNG sequence
            output_composition=output_composition,  # Output path for composition
            output_video_mbps=4,  # Output video bitrate (only for video)
            downsample_ratio=None,  # Auto downsampling
            seq_chunk=24  # Process 24 frames in parallel
        )

    # Apply chroma key replacement to the video
    chroma_replace(output_video, full_background_path, phone_background_path, args.zoom_scale, args.zoom_increment, args.output_video_name, lower_green, upper_green, lower_blue, upper_blue)

if __name__ == "__main__":
    main()
