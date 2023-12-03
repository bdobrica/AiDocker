from pathlib import Path
from typing import Generator, List, Tuple

import keras
import numpy as np
from PIL import Image

from daemon import AiBatch as Batch


class AiBatch(Batch):
    @property
    def _merged_buffer(self) -> np.ndarray:
        sizes = [image.size for _, image in self._buffer]
        self._max_size = (
            max([width for width, _ in sizes]),
            max([height for _, height in sizes]),
        )

        return np.stack(map(self._image_to_array, self._buffer), axis=0)

    def _image_to_array(self, image: Image.Image) -> np.ndarray:
        if image.size != self._max_size:
            image = image.resize(size=self._max_size)
        return keras.utils.img_to_array(image)

    def get_image_item(self) -> Generator[Image.Image, None, None]:
        for source_file in self.source_files:
            yield Image.open(source_file)

    def prepare(self, batch_size: int) -> Generator[np.ndarray, None, None]:
        """
        The method checks the file queue and packs together <batch_size> images into a numpy array that's used as input
        for the machine learning model.
        :param batch_size: Number of images that are packed together into a batch
        :return: Generator of numpy arrays that are used as input for the machine learning model in (N, H, W, C) format.
        """
        self._buffer: List[Image.Image] = []

        for item in self.get_image_item():
            self._buffer.append(item)
            if len(self._buffer) >= batch_size:
                yield self._merged_buffer
                self._buffer = []
        if self._buffer:
            yield self._merged_buffer

    def serve(self, output: np.ndarray) -> None:
        """
        The method is called for each batch of output data. The output data is a numpy array of shape (N, H, W, C).
        The output data is converted to images and saved to the prepared folder. The metadata is updated to reflect
        that the file has been processed. The source file is removed from the queue.
        :param output: Output data produced by the machine learning model.
        """
        for image, output_image in zip(self._buffer, output):
            source_file = Path(image.filename)
            output_image = Image.fromarray(output_image)
            if output_image.size != image.size:
                output_image = output_image.resize(size=image.size)

            output_image.save(self.prepared_path / source_file.name)
            self._update_metadata(self.get_metadata_file_path(image), {"processed": "true"})
            source_file.unlink()
