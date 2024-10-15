import numpy as np
import os

class EditorConfig:
    """
    Attributes:
        original_video (str): Path to the original video file.
        full_background (str): Path to the full background video.
        phone_background (str): Path to the phone background video (optional).
        model_path (str): Path to the model.
        zoom_scale (float): Initial zoom scale.
        zoom_increment (float): Zoom increment value.
        output_video_name (str): Name of the output video file.
        lower_blue (np.ndarray): Lower HSV threshold for blue color.
        upper_blue (np.ndarray): Upper HSV threshold for blue color.
        lower_green (np.ndarray): Lower HSV threshold for green color.
        upper_green (np.ndarray): Upper HSV threshold for green color.
        output_type (str): Output type ("png" or "video").
        robust_model (str): Model type (resnet50 or mobilenetv3).
        replace_output_video (str): Path to the resulting video from replace_color
        output_composition (str): Path to AI-processed video
        processing_model (str): Robust file processing model (cpu or gpu (cuda))
    """

    original_video: str = ''
    full_background: str = ''
    phone_background: str = ''
    model_path: str = 'rvm_resnet50.pth'
    zoom_scale: float = 0.2
    zoom_increment: float = 1
    output_video_name: str = 'output_video'
    lower_blue: np.ndarray = np.array([80, 50, 80])
    upper_blue: np.ndarray = np.array([130, 255, 255])
    lower_green: np.ndarray = np.array([35, 40, 40])
    upper_green: np.ndarray = np.array([85, 255, 255])
    robust_output_type: str = 'png'
    model_type: str = 'resnet50'
    replace_output_video: str = ''
    output_composition: str = ''
    processing_model: str = 'cpu'

    def __init__(self, args=None):
        """
        Initialize parameters with the option to overwrite them using command-line arguments.

        Args:
            args: The object returned by parse_args(), containing command-line arguments.
        """
        # If the args object is passed, overwrite the default values
        if args:
            self.original_video = getattr(args, 'original_video', self.original_video)
            self.full_background = getattr(args, 'full_background', self.full_background)
            self.phone_background = getattr(args, 'phone_background', self.phone_background)
            self.model_path = getattr(args, 'model_path', self.model_path)
            self.zoom_scale = getattr(args, 'zoom_scale', self.zoom_scale)
            self.zoom_increment = getattr(args, 'zoom_increment', self.zoom_increment)
            self.output_video_name = getattr(args, 'output_video_name', self.output_video_name)

            # HSV ranges
            self.lower_blue = np.array(getattr(args, 'lower_blue', self.lower_blue))
            self.upper_blue = np.array(getattr(args, 'upper_blue', self.upper_blue))
            self.lower_green = np.array(getattr(args, 'lower_green', self.lower_green))
            self.upper_green = np.array(getattr(args, 'upper_green', self.upper_green))

            self.robust_output_type = getattr(args, 'robust_output_type', self.robust_output_type)
            self.model_type = getattr(args, 'robust_model', self.model_type)
            self.processing_model = getattr(args, 'processing_model', self.processing_model)

            self.replace_output_video = f'{os.path.splitext(os.path.basename(self.original_video))[0]}_replace_color.mp4'
            self.output_composition = f'./robust/{os.path.splitext(self.original_video)[0]}_output_{self.robust_output_type}'