import os
import cv2
from iterators.base_iterator import BaseIterator

class FrameIterator(BaseIterator):

    def __init__(self, path):
        self._file_path = path
        self._current_index = 0

        self.file_pattern = '{:04d}.png'
        self.files = [f for f in os.listdir(self._file_path) if f.endswith('.png')]
        self.num_files = len(self.files)


    def __iter__(self):
        return self

    def __next__(self):
        if self._current_index >= self.num_files:
            raise StopIteration

        file_name = self.file_pattern.format(self._current_index)
        file_path = os.path.join(self._file_path, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Image {file_name} not found, skipping...")
            self._current_index += 1
            return self.__next__()

        self._current_index += 1
        return image
