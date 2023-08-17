#!/usr/bin/env python3
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
        self.prefix = ""

    def get_text_item(self) -> Generator[TextItem, None, None]:
        for source_file in self.source_files:
            search_space = self.get_metadata(source_file).get("search_space", "")
            yield from read_text(source_file, search_space)
            # TODO: need to check if this update is really in the right place
            self._update_metadata(source_file, {"processed": "true"})

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
            item.store(self.redis, vector.flatten(), self.prefix)
