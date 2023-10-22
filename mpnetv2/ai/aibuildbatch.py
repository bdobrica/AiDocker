#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Union

from redis import Redis
from torch import Tensor

from daemon import AiBatch as Batch
from daemon import PathLike

from .reader import TextItem, read_text


class AiBuildBatch(Batch):
    def __init__(self, staged_files: Union[PathLike, Iterable[PathLike]], redis: Redis) -> None:
        """
        Queue of documents to be processed. The documents are converted into text items which in turn are passed through
        a machine learning based on the MPNet v2 model. The model output is stored in Redis.
        :param staged_files: List of documents to be processed
        :param redis: Redis connection
        """
        super().__init__(staged_files=staged_files)
        self.redis = redis
        self._remove_items()
        self._finished = []

    def _remove_item(self, source_file: Path) -> None:
        """
        Method to remove a document from the Redis database. The document is identified by the search space and the
        document name.
        :param source_file: Path to the document, e.g. "path/to/document.pdf"
        """
        search_space = self.get_metadata(source_file).get("search_space", "")
        pieces = [os.getenv("DOC_PREFIX", "doc"), search_space, source_file.stem]
        prefix = ":".join([piece for piece in pieces if piece])
        removed_items = 0
        self.set_metadata(source_file, {"processed": "false", "state": "deleting"})
        for key in self.redis.scan_iter(f"{prefix}:*"):
            self.redis.delete(key)
            removed_items += 1
        self.set_metadata(source_file, {"processed": "true", "state": "done", "removed_items": removed_items})
        source_file.unlink()
        self.source_files.remove(source_file)

    def _remove_items(self):
        """
        Method to remove documents from the Redis database. Calls the _remove_item method for each document.
        Documents that are marked for deletion have the suffix ".delete".
        """
        to_remove = [source_file for source_file in self.source_files if source_file.suffix == ".delete"]
        if not to_remove:
            return
        for source_file in to_remove:
            self._remove_item(source_file)

    def get_text_item(self) -> Generator[TextItem, None, None]:
        """
        For each document, the method converts the document into a list of text items. The text items are yielded one
        by one.
        :return: Generator of text items of all documents
        """
        for source_file in self.source_files:
            search_space = self.get_metadata(source_file).get("search_space", "")
            self.set_metadata(source_file, {"processed": "false", "state": "processing"})
            yield from read_text(source_file, search_space)
            # TODO: need to check if this update is really in the right place
            self._finished.append(source_file)

    def prepare(self, batch_size: int) -> Generator[List[TextItem], None, None]:
        """
        The method converts the generator of text items into batches of text items. The batch size is defined by the
        batch_size parameter.
        :param batch_size: Batch size of the text items that are passed through the machine learning model
        :return: Generator of batches of text items
        """
        self._buffer: List[TextItem] = []
        for item in self.get_text_item():
            self._buffer.append(item)
            if len(self._buffer) >= batch_size:
                yield self._buffer
                self._buffer = []
        if self._buffer:
            yield self._buffer

    def serve(self, model_output: Optional[Tensor]) -> None:
        """
        The method processes the model output and stores it in Redis. The model output is a tensor of vectors.
        :param model_output: Model output as tensor
        """
        if model_output is None:
            # TODO: should do some update that processing failed
            return

        model_output = model_output.cpu().numpy()
        for item, vector in zip(self._buffer, model_output):
            item: TextItem
            vector: Tensor
            item.store(self.redis, vector.flatten())

        while self._finished:
            source_file = self._finished.pop()
            self.set_metadata(source_file, {"processed": "true", "state": "done"})
