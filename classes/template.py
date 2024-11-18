from typing import Dict
import numpy as np

class Template:
    def __init__(self, template_path: str, resize: Dict[str, int], threshold: float = 0.8, background_hex_color: str = '',  background_color_hsv_range: Dict[str, np.ndarray] = None, form_factor: str = '', form_size: int = 10, radius_raising: bool = False, template_skip_frames: int = 0):
        self.template_path = template_path
        self.resize = resize
        self.threshold = threshold
        self.background_hex_color = background_hex_color
        self.background_color_hsv_range = background_color_hsv_range
        self.form_factor = form_factor
        self.form_size = form_size
        self.radius_raising = radius_raising
        self.template_skip_frames = template_skip_frames

        # Инициализация закрытых атрибутов состояния
        self._zoom_factor = 1.0
        self._darkness = 0.0
        self._zoom_direction = 1
        self._completed = False
        self.best_match = None
        self.best_val = 0
        self.first_initial = True

    @property
    def zoom_factor(self):
        return self._zoom_factor

    @zoom_factor.setter
    def zoom_factor(self, value):
        self._zoom_factor = value

    @property
    def darkness(self):
        return self._darkness

    @darkness.setter
    def darkness(self, value):
        self._darkness = value

    @property
    def zoom_direction(self):
        return self._zoom_direction

    @zoom_direction.setter
    def zoom_direction(self, value):
        self._zoom_direction = value

    @property
    def completed(self):
        return self._completed

    @completed.setter
    def completed(self, value):
        self._completed = value
