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
        super().__init__(staged_files=staged_files)
        self.redis = redis
        self._remove_items()

    def _remove_item(self, source_file: Path) -> None:
        search_space = self.get_metadata(source_file).get("search_space", "")
        pieces = [os.getenv("DOC_PREFIX", "doc"), search_space, source_file.stem]
        prefix = ":".join([piece for piece in pieces if piece])
        removed_items = 0
        self._update_metadata(source_file, {"processed": "false", "state": "deleting"})
        for key in self.redis.scan_iter(f"{prefix}:*"):
            self.redis.delete(key)
            removed_items += 1
        self._update_metadata(source_file, {"processed": "true", "state": "done", "removed_items": removed_items})
        source_file.unlink()
        self.source_files.remove(source_file)

    def _remove_items(self):
        to_remove = [source_file for source_file in self.source_files if source_file.suffix == ".delete"]
        if not to_remove:
            return
        for source_file in to_remove:
            self._remove_item(source_file)

    def get_text_item(self) -> Generator[TextItem, None, None]:
        for source_file in self.source_files:
            search_space = self.get_metadata(source_file).get("search_space", "")
            self._update_metadata(source_file, {"processed": "false", "state": "processing"})
            yield from read_text(source_file, search_space)
            # TODO: need to check if this update is really in the right place
            self._update_metadata(source_file, {"processed": "true", "state": "done"})

    def prepare(self, batch_size: int) -> Generator[List[TextItem], None, None]:
        self._buffer: List[TextItem] = []
        for item in self.get_text_item():
            self._buffer.append(item)
            if len(self._buffer) >= batch_size:
                yield self._buffer
                self._buffer = []
        if self._buffer:
            yield self._buffer

    def serve(self, model_output: Optional[Tensor]) -> None:
        if model_output is None:
            # TODO: should do some update that processing failed
            return

        model_output = model_output.cpu().numpy()
        for item, vector in zip(self._buffer, model_output):
            item: TextItem
            vector: Tensor
            item.store(self.redis, vector.flatten())
