"""
TextItem is an object that can be easily stored in Redis and searched with RedisSearch. It contains the text content of
a document, the page and paragraph number, the path to the document and can be converted into a vector with a
transformer model.
"""
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Generator, List, Optional, Union

import numpy as np
from redis import Redis
from redis.commands.search.document import Document
from redis.commands.search.query import Query


class TextItem:
    def __init__(
        self,
        text: str,
        token: str = "",
        search_space: str = "",
        page: Optional[int] = None,
        paragraph: Optional[int] = None,
        path: Optional[Union[str, Path]] = None,
        score: Optional[float] = None,
    ):
        """
        Constructor of TextItem
        :param text: Text content of the text item. Usually a paragraph of a document. The only required parameter.
        :param token: Unique identifier of the document, e.g. the filename without the suffix. Part of redis key
        :param search_space: Documents can be grouped into different search spaces, e.g. "en" or "de"
        :param page: Page number of the text item
        :param paragraph: Paragraph number of the text item
        :param path: Path to the document, e.g. "path/to/document.pdf"
        :param score: Score of the text item, e.g. the cosine similarity of the vector of the text item and the query
        """
        self.text = re.sub(r"\s+", " ", text).strip()
        self.token = token
        self.search_space = search_space
        self.page = page
        self.paragraph = paragraph
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.score = score

    def __str__(self):
        """When cast to string, the text content is returned"""
        return self.text

    def __repr__(self):
        """The representation of the object contains its important attributes"""
        return f"TextItem(text='{self.text}', page={self.page}, paragraph={self.paragraph}, path={self.path})"

    def _dup(self, text: str = None):
        """Duplicates the object with a new text content. Used for splitting the text item into smaller pieces."""
        return TextItem(
            text=text or self.text,
            search_space=self.search_space,
            page=self.page,
            paragraph=self.paragraph,
            path=self.path,
        )

    @property
    def prefix(self) -> str:
        """
        Prefix of the redis key. Can be used to group documents into different search spaces.
        It contains $DOC_PREFIX or "doc", the search space and the token.
        """
        pieces = [os.getenv("DOC_PREFIX", "doc"), self.search_space, self.token]
        return ":".join([piece for piece in pieces if piece])

    @property
    def key(self) -> str:
        """
        Unique identifier of the text item. Used as redis key.
        It's made up of the prefix and the sha256 hash of the text content.
        """
        digest = sha256(self.text.encode("utf-8")).hexdigest()
        return f"{self.prefix}:{digest}"

    def asdict(self):
        """Converts the object into a dictionary. Used to store the object in Redis."""
        return {
            "text": self.text,
            "search_space": self.search_space,
            "page": self.page if self.page is not None else -1,
            "paragraph": self.paragraph if self.paragraph is not None else -1,
            "path": self.path.as_posix() or "",
        }

    def split(self, max_length: int = 512, overlap: int = 0) -> Generator["TextItem", None, None]:
        """
        Splits the text item into smaller pieces. The text content is split at the last space before the maximum length.
        :param max_length: Maximum length of a text item content, usually matches the maximum input length of the model.
        :param overlap: Overlap between the text content of two consecutive text items. Default: 0
        :return: Generator of text items
        """
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
        """
        Stores the text item in Redis. The text content is stored as a redis hash. The vector is stored as a redis
        tensor.
        :param redis: Redis connection
        :param vector: Vector of the text item
        """
        redis.hset(
            self.key,
            mapping={
                **self.asdict(),
                "vector": vector.astype(np.float32).tobytes(),
            },
        )

    def match(self, redis: Redis, vector: np.ndarray, docs: int = 10) -> List[Document]:
        """
        Searches for similar text items in Redis. The vector of the text item is used to search for similar vectors.
        :param redis: Redis connection
        :param vector: Vector of the text item
        :param docs: Number of documents to return
        :return: List of redis Document objects
        """
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
