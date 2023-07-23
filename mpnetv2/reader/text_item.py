import re
from hashlib import sha256
from typing import Generator, Optional

import numpy as np
from redis import Redis

# C:\Users\bdobr\AppData\Roaming\nltk_data


class TextItem:
    def __init__(self, text: str, page: Optional[int], paragraph: Optional[int], path: Optional[str]):
        self.text = re.sub(r"\s+", " ", text).strip()
        self.page = page
        self.paragraph = paragraph
        self.path = path

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"TextItem(text='{self.text}', page={self.page}, paragraph={self.paragraph}, path={self.path})"

    def _dup(self, text: str = None):
        return TextItem(
            text=text or self.text,
            page=self.page,
            paragraph=self.paragraph,
            path=self.path,
        )

    @property
    def key(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()

    def asdict(self):
        return {
            "text": self.text,
            "page": self.page if self.page is not None else -1,
            "paragraph": self.paragraph if self.paragraph is not None else -1,
            "path": self.path or "",
        }

    def split(self, max_length: int = 512, overlap: int = 0) -> Generator["TextItem"]:
        if len(self.text) <= max_length:
            yield self
            return

        start = 0
        while True:
            if start + max_length + 1 > len(self.text):
                yield self._dup(text=self.text[start:])
                break
            sub_text = self.text[start : start + max_length + 1]
            pos = sub_text.rindex(" ")
            yield self._dup(text=sub_text[:pos])
            if overlap:
                pos = sub_text[:-overlap].rindex(" ")
            start = start + pos

    def store(self, redis: Redis, vector: np.ndarray) -> None:
        redis.hset(
            self.key,
            mapping={
                **self.asdict(),
                "vector": vector.tobytes(),
            },
        )
