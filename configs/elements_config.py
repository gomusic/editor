class ElementsConfig:
    """
        Attributes:
            skip_frames (int): Number of frames to skip before processing starts.
            zoom_speed (float): Speed at which the zoom effect is applied.
            max_zoom_factor (int): Maximum zoom factor for the zoom effect.
            threshold (float): Minimum matching value for template matching.
    """

    def __init__(self):
        self.skip_frames = 204 # 33-42
        self.zoom_speed = 1
        self.max_zoom_factor = 5