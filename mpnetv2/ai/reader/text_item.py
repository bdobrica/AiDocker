import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Generator, List, Optional, Union

import numpy as np
from redis import Redis
from redis.commands.search.document import Document
from redis.commands.search.query import Query

# C:\Users\bdobr\AppData\Roaming\nltk_data


class TextItem:
    def __init__(
        self,
        text: str,
        search_space: str = "",
        page: Optional[int] = None,
        paragraph: Optional[int] = None,
        path: Optional[Union[str, Path]] = None,
        score: Optional[float] = None,
    ):
        self.text = re.sub(r"\s+", " ", text).strip()
        self.search_space = search_space
        self.page = page
        self.paragraph = paragraph
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.score = score

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"TextItem(text='{self.text}', page={self.page}, paragraph={self.paragraph}, path={self.path})"

    def _dup(self, text: str = None):
        return TextItem(
            text=text or self.text,
            search_space=self.search_space,
            page=self.page,
            paragraph=self.paragraph,
            path=self.path,
        )

    @property
    def prefix(self) -> str:
        pieces = [os.getenv("DOC_PREFIX", "doc"), self.search_space]
        return ":".join([piece for piece in pieces if piece])

    @property
    def key(self) -> str:
        digest = sha256(self.text.encode("utf-8")).hexdigest()
        return f"{self.prefix}:{digest}"

    def asdict(self):
        return {
            "text": self.text,
            "search_space": self.search_space,
            "page": self.page if self.page is not None else -1,
            "paragraph": self.paragraph if self.paragraph is not None else -1,
            "path": self.path.as_posix() or "",
        }

    def split(self, max_length: int = 512, overlap: int = 0) -> Generator["TextItem", None, None]:
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
                "vector": vector.astype(np.float32).tobytes(),
            },
        )

    def match(self, redis: Redis, vector: np.ndarray, docs: int = 10) -> List[Document]:
        query = (
            Query(f"*=>[KNN {docs} @vector $vec as score]")
            .sort_by("score")
            .return_fields("text", "page", "paragraph", "path", "score")
            .dialect(2)
        )

        query_params = {"vec": vector.flatten().astype(np.float32).tobytes()}
        return (
            redis.ft(self.prefix)
            .search(
                query=query,
                query_params=query_params,
            )
            .docs
        )
