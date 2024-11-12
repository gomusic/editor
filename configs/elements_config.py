class ElementsConfig:
    """
        Attributes:
            skip_frames (int): Number of frames to skip before processing starts.
            zoom_speed (float): Speed at which the zoom effect is applied.
            max_zoom_factor (int): Maximum zoom factor for the zoom effect.
            contours_threshold (float): % of largest contour area, contours are sorted by size, 0.7 - 70% of largest area and above.
    """

    def __init__(self):
        self.skip_frames = 175 # 33-42
        self.zoom_speed = 0.5
        self.darkening_speed = 0.3
        self.max_zoom_factor = 5
        self.contours_threshold = 1