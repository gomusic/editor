import os
import cv2
from iterators.base_iterator import BaseIterator

class VideoIterator(BaseIterator):

    def __init__(self, path):
        self._file_path = path
        self._current_index = 0

        self.video_path = os.path.join(self._file_path, 'com.mp4')
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"Video file {self.video_path} not found or couldn't be opened.")

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame
