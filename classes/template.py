from typing import Dict
import numpy as np

class Template:
    def __init__(self, path: str, resize: Dict[str, int], threshold = 0.8):
        self.path = path
        self.resize = resize
        self.threshold = threshold

        # Инициализация закрытых атрибутов состояния
        self._zoom_factor = 1.0
        self._darkness = 0.0
        self._zoom_direction = 1
        self._completed = False
        self.best_match = 0
        self.best_val = 0

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
