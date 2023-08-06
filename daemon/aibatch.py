import json
import os
import time
from pathlib import Path
from typing import Iterable, Union

from .daemon import PathLike


class AiBatch:
    SOURCE_PATH = Path(os.getenv("SOURCE_PATH", "/tmp/ai/source"))
    PREPARED_PATH = Path(os.getenv("PREPARED_PATH", "/tmp/ai/prepared"))
    DEFAULT_EXTENSION = os.getenv("DEFAULT_EXTENSION", "png").lower()

    @staticmethod
    def _load_metadata(meta_file: Path) -> dict:
        if not meta_file.is_file():
            return {}
        with open(meta_file, "r") as fp:
            return json.load(fp)

    @staticmethod
    def _update_metadata(meta_file: Path, data: dict) -> None:
        metadata = AiBatch._load_metadata(meta_file)
        if "update_time" not in metadata:
            metadata["update_time"] = time.time()
        metadata.update(data)
        with open(meta_file, "w") as fp:
            json.dump(metadata, fp)

    @staticmethod
    def get_queued_files() -> list:
        return list(AiBatch.SOURCE_PATH.iterdir())

    @staticmethod
    def get_queue_size() -> int:
        return len(AiBatch.get_queued_files())

    def _move_staged_files(self) -> None:
        _ = [staged_file.rename(source_file) for staged_file, source_file in zip(self.staged_files, self.source_files)]

    def __init__(self, staged_files: Union[PathLike, Iterable[PathLike]]):
        if isinstance(staged_files, str):
            staged_files = Path(staged_files)
        if isinstance(staged_files, Path):
            staged_files = [staged_files]

        self.staged_files = list(staged_files)
        self.meta_files = [staged_file.with_suffix(".json") for staged_file in self.staged_files]
        self.source_files = [self.SOURCE_PATH / staged_file.name for staged_file in self.staged_files]
        self.metadata = [AiBatch._load_metadata(meta_file) for meta_file in self.meta_files]
        self.prepared_files = [
            self.PREPARED_PATH
            / (staged_file.stem + "." + self.metadata[index].get("output_extension", self.DEFAULT_EXTENSION))
            for index, staged_file in enumerate(self.staged_files)
        ]
        self._move_staged_files()

    def prepare(self) -> any:
        raise NotImplementedError("You must implement prepare() -> any")

    def serve(self, inference_data: any) -> None:
        raise NotImplementedError("You must implement serve(inference_data: any)")

    def update_metadata(self, data: dict) -> None:
        _ = [AiBatch._update_metadata(meta_file, data) for meta_file in self.meta_files]

    def __del__(self):
        _ = [source_file.unlink() for source_file in self.source_files]
