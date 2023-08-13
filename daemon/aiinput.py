import os
from pathlib import Path
from typing import Any, Optional

from .daemon import PathLike
from .filequeuemixin import FileQueueMixin


class AiInput(FileQueueMixin):
    DEFAULT_EXTENSION = os.getenv("DEFAULT_EXTENSION", "png").lower()

    def _move_staged_file(self) -> None:
        self.staged_file.rename(self.source_file)

    def __init__(self, staged_file: PathLike):
        if isinstance(staged_file, str):
            staged_file = Path(staged_file)
        self.staged_file = staged_file
        self.meta_file = self.staged_file.with_suffix(".json")
        self.source_file = self.SOURCE_PATH / self.staged_file.name
        self.metadata = self._load_metadata(self.meta_file)
        self.prepared_file = self.PREPARED_PATH / (
            self.staged_file.stem + "." + self.metadata.get("output_extension", self.DEFAULT_EXTENSION)
        )
        self._move_staged_file()

    def prepare(self) -> Any:
        raise NotImplementedError("You must implement prepare() -> Any")

    def serve(self, inference_data: Optional[Any]) -> None:
        raise NotImplementedError("You must implement serve(inference_data: Any)")

    def update_metadata(self, data: dict) -> None:
        AiInput._update_metadata(self.meta_file, data)

    def __del__(self):
        self.source_file.unlink()
