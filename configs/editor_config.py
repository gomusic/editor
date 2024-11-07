import numpy as np
import os

class EditorConfig:
    """
        Attributes:
            original_video (str): Path to the original video file.
            full_background (str): Path to the full background video.
            phone_background (str): Path to the phone background video.
            model_path (str): Path to the Robust Video Matting model.
            zoom_scale (float): Initial zoom scale for the dynamic zoom effect.
            zoom_increment (float): Increment applied to the zoom per frame.
            output_video_name (str): Name of the output video file (without extension).
            lower_blue (np.ndarray): Lower HSV threshold for blue color in chroma key processing.
            upper_blue (np.ndarray): Upper HSV threshold for blue color in chroma key processing.
            lower_green (np.ndarray): Lower HSV threshold for green color in chroma key processing.
            upper_green (np.ndarray): Upper HSV threshold for green color in chroma key processing.
            robust_output_type (str): Robust Video Matting output type (either "png" for image sequence or "video" for a full video).
            robust_model (str): Robust Video Matting model type (either resnet50 or mobilenetv3).
            replace_output_video (str): Path to the video output after color replacement is applied.
            output_composition (str): Path to the final AI-processed video composition.
            processing_model (str): File processing model for Robust Video Matting ('cpu' or 'gpu' (cuda)).
    """

    original_video: str = ''
    full_background: str = ''
    phone_background: str = ''
    model_path: str = '../rvm_resnet50.pth'
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

        # TODO: можно сделать return сразу если аргументов нет, в твоем случае возможен запуск без аргументов?
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
            # TODO: если в self.original_video будет передан путь типа ./ - все сломается, если передать /home/video.test - все сломается
            self.output_composition = f'./robust/{os.path.splitext(self.original_video)[0]}_output_{self.robust_output_type}'