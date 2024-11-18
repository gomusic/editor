import numpy as np
import os


class EditorConfig:
    """
    Class for storing editor settings.

    Attributes:
        original_video (str): Path to the original video file.
        full_background (str): Path to the full background video.
        phone_background (str): Path to the phone background video.
        model_path (str): Path to the Robust Video Matting model.
        zoom_scale (float): Initial zoom scale for the dynamic zoom effect.
        zoom_increment (float): Zoom increment per frame.
        output_video_name (str): Name of the output video (without extension).
        lower_blue (np.ndarray): Lower HSV threshold for blue color during processing.
        upper_blue (np.ndarray): Upper HSV threshold for blue color.
        lower_green (np.ndarray): Lower HSV threshold for green color.
        upper_green (np.ndarray): Upper HSV threshold for green color.
        robust_output_type (str): RVM output type (png for image sequence or video for full video).
        processing_model (str): Processing type for RVM (cpu or gpu).
        output_dir (str): Main directory for saving results.
        start_phone_video (bool): Whether to start the clip immediately after the phone screen appears in the frame.
    """

    original_video = ''
    full_background = ''
    phone_background = ''
    model_path = '../rvm_resnet50.pth'
    zoom_scale = 0.2
    zoom_increment = 1
    output_video_name = 'output_video'
    lower_blue = np.array([80, 50, 80])
    upper_blue = np.array([130, 255, 255])
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    robust_output_type = 'png'
    processing_model = 'cpu'
    output_dir = './results'  # Default directory
    start_phone_video = False


    def __init__(self, args=None):

        if args:
            self.original_video = getattr(args, 'original_video', self.original_video)
            if not os.path.isfile(self.original_video):
                raise ValueError(f"Invalid path provided for original_video: {self.original_video}")

            self.full_background = getattr(args, 'full_background', self.full_background)
            self.phone_background = getattr(args, 'phone_background', self.phone_background)
            self.model_path = getattr(args, 'model_path', self.model_path)
            self.zoom_scale = getattr(args, 'zoom_scale', self.zoom_scale)
            self.zoom_increment = getattr(args, 'zoom_increment', self.zoom_increment)
            self.output_video_name = getattr(args, 'output_video_name', self.output_video_name)
            self.lower_blue = np.array(getattr(args, 'lower_blue', self.lower_blue))
            self.upper_blue = np.array(getattr(args, 'upper_blue', self.upper_blue))
            self.lower_green = np.array(getattr(args, 'lower_green', self.lower_green))
            self.upper_green = np.array(getattr(args, 'upper_green', self.upper_green))
            self.robust_output_type = getattr(args, 'robust_output_type', self.robust_output_type)
            self.processing_model = getattr(args, 'processing_model', self.processing_model)
            self.output_dir = getattr(args, 'output_dir', self.output_dir)  # Directory specified by the user
            self.start_phone_video = getattr(args, 'start_phone_video', self.start_phone_video)

        # Extract the filename without extension
        self.filename_without_extension = os.path.splitext(os.path.basename(self.original_video))[0]
        self.main_folder_path = os.path.join(self.output_dir, self.filename_without_extension)

        # Paths considering the folder structure
        self.replace_output_video_folder_path = os.path.join(self.main_folder_path, 'replace')
        self.replace_output_video_path = os.path.join(self.main_folder_path, 'replace', f'{self.filename_without_extension}_replace_color.mp4')
        self.output_composition_path = os.path.join(self.main_folder_path, 'robust', f'{self.filename_without_extension}_{self.robust_output_type}')
        self.temp_dir = os.path.join(self.main_folder_path, 'temp')
        self.output_video_path = os.path.join(self.main_folder_path, f'{self.output_video_name}.mp4')

        # Directory for intermediate RVM output
        self.robust_output_dir = os.path.join(self.output_composition_path)

        # Directories for backgrounds
        self.phone_background_dir = os.path.join(self.temp_dir, 'background_phone_video')
        self.full_background_dir = os.path.join(self.temp_dir, 'background_video')
