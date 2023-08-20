import os
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

from .daemon import PathLike
from .filequeuemixin import FileQueueMixin


class AiBatch(FileQueueMixin):
    DEFAULT_EXTENSION = os.getenv("DEFAULT_EXTENSION", "png").lower()

    def _move_staged_files(self) -> None:
        _ = [staged_file.rename(self.source_path / staged_file.name) for staged_file in self.staged_files]
        self.source_files = [
            self.source_path / staged_file.name
            for staged_file in self.staged_files
            if (self.source_path / staged_file.name).is_file()
        ]

    def __init__(self, staged_files: Union[PathLike, Iterable[PathLike]]):
        if isinstance(staged_files, (str, Path)):
            staged_files = [staged_files]
        self.staged_files = [Path(staged_file) for staged_file in staged_files if Path(staged_file).is_file()]
        self.meta_files = [staged_file.with_suffix(".json") for staged_file in self.staged_files]
        self.metadata = [AiBatch._load_metadata(meta_file) for meta_file in self.meta_files]
        self.source_files = []
        self._move_staged_files()

    def prepare(self) -> Any:
        raise NotImplementedError("You must implement prepare() -> Any")

    def serve(self, inference_data: Optional[Any]) -> None:
        raise NotImplementedError("You must implement serve(inference_data: Any)")

    def update_metadata(self, data: dict) -> None:
        _ = [AiBatch._update_metadata(meta_file, data) for meta_file in self.meta_files]

    def __del__(self):
        _ = [source_file.unlink() for source_file in self.source_files]
