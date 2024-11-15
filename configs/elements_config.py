class ElementsConfig:
    """
        Attributes:
            skip_frames (int): Number of frames to skip before processing starts.
            zoom_speed (float): Speed at which the zoom effect is applied.
            max_zoom_factor (int): Maximum zoom factor for the zoom effect.
            contours_threshold (float): % of largest contour area, contours are sorted by size, 0.7 - 70% of largest area and above.
    """
    max_zoom_factor = 5

    def __init__(self, fps=25):
        self.skip_frames = 196 # 33-42
        self.zoom_duration = 2 # Zoom time in seconds
        self._fps = fps
        self.zoom_speed = self.max_zoom_factor / (self.zoom_duration * self._fps)
        self.darkening_speed = 0.3
        self.contours_threshold = 1
        self.radius_increase = 10 # The value of how much the radius will be larger than the element

        self._start_skipping = False

    @property
    def start_skipping(self):
        return self._start_skipping


    @start_skipping.setter
    def start_skipping(self, value):
        self._start_skipping = value


    @property
    def fps(self):
        return self._fps


    @fps.setter
    def fps(self, value):
        if value <= 0:
            raise ValueError("FPS must be greater than 0.")
        self._fps = value
        self._update_zoom_speed()


    def _update_zoom_speed(self):
        """Updates the zoom speed based on the current FPS and zoom duration."""
        self.zoom_speed = self.max_zoom_factor / (self.zoom_duration * self._fps)