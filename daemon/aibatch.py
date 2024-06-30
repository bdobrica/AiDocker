"""
File queue batch extraction and processing. Intended to be used by the AiBatchDaemon.
"""

import os
from pathlib import Path
from typing import Any, Iterable, Optional

from .daemon import PathLike
from .filequeuemixin import FileQueueMixin


class AiBatch(FileQueueMixin):
    DEFAULT_EXTENSION = os.getenv("DEFAULT_EXTENSION", "png").lower()

    def _move_staged_files(self) -> None:
        """
        Move the staged files from the staged folder to the source folder.
        """
        _ = [staged_file.rename(self.source_path / staged_file.name) for staged_file in self.staged_files]
        self.source_files = [
            self.source_path / staged_file.name
            for staged_file in self.staged_files
            if (self.source_path / staged_file.name).is_file()
        ]
        for source_file in self.source_files:
            self.set_metadata(source_file, {"processed": "false", "state": "queued"})

    def __init__(self, staged_files: Iterable[PathLike]) -> None:
        """
        Initialize the batch.
        :param staged_files: The files that are part of the current batch.
        """
        self.staged_files = [Path(staged_file) for staged_file in staged_files if Path(staged_file).is_file()]
        self.meta_files = [staged_file.with_suffix(".json") for staged_file in self.staged_files]
        self.metadata = [AiBatch._load_metadata(meta_file) for meta_file in self.meta_files]
        self.source_files = []
        self._move_staged_files()

    def prepare(self) -> Any:
        """
        Prepare the batch for processing. This method is responsible for loading the data from the source files and
        returning it in a format that is suitable for AI processing (probably numpy or pytorch.Tensor).
        :return: The data to be processed.
        """
        raise NotImplementedError("You must implement prepare() -> Any")

    def serve(self, inference_data: Optional[Any]) -> None:
        """
        Get's the model output and prepares it by creating the output files under the prepared folder.
        :param inference_data: The data returned by the AI model.
        """
        raise NotImplementedError("You must implement serve(inference_data: Any)")

    def update_metadata(self, data: dict) -> None:
        """
        Helper method to update the metadata for all the files in the batch.
        :param data: The data to update the metadata with.
        """
        _ = [AiBatch._update_metadata(meta_file, data) for meta_file in self.meta_files]

    def __del__(self):
        """
        Cleanup the source files.
        """
        _ = [source_file.unlink() for source_file in self.source_files]
