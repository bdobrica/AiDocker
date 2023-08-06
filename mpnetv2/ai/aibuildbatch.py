#!/usr/bin/env python3
import itertools
import os
from pathlib import Path
from typing import Generator, List

import numpy as np
from redis import Redis

from daemon import AiBatch as Batch

from .reader import TextItem, read_text


class AiBuildBatch(Batch):
    def set_redis_connection(self, redis: Redis) -> None:
        self.redis = redis

    def get_text_item(self) -> Generator[TextItem, None, None]:
        for source_file in self.source_files:
            yield from read_text(Path(source_file))

    def prepare(self) -> Generator[List[TextItem], None, None]:
        yield list(itertools.islice(self.get_text_item(), 0, int(os.getenv("BATCH_SIZE", 8))))
