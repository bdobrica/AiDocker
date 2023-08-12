"""
This module implements a simple file-queue. It is used to pass files between the different stages of the pipeline.
The file-queue is implemented as a directory structure, where each directory represents a stage in the pipeline.
The files are moved between the directories as they progress through the pipeline.

There are three directories:
    - STAGED_PATH: The directory where the files are placed when they are staged for processing.
    - SOURCE_PATH: The directory where the files are placed when they are queued for processing.
    - PREPARED_PATH: The directory where the files are placed when they were already processed.

The files are moved between the directories as follows:
    - STAGED_PATH -> SOURCE_PATH: When the file is staged for processing, it is moved from the STAGED_PATH to the
        SOURCE_PATH.
    - SOURCE_PATH -> PREPARED_PATH: When the file is queued for processing, it is moved from the SOURCE_PATH to the
        PREPARED_PATH.

Every file has a corresponding metadata file. The metadata file is a JSON file with the same name as the file, but with
the .json extension. The metadata file contains information about the file, such as the time it was last updated.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional


class FileQueue:
    STAGED_PATH = Path(os.getenv("STAGED_PATH", "/tmp/ai/staged"))
    SOURCE_PATH = Path(os.getenv("SOURCE_PATH", "/tmp/ai/source"))
    PREPARED_PATH = Path(os.getenv("PREPARED_PATH", "/tmp/ai/prepared"))

    @staticmethod
    def _load_metadata(meta_file: Path) -> dict:
        if not meta_file.is_file():
            return {}
        with open(meta_file, "r") as fp:
            return json.load(fp)

    @staticmethod
    def _update_metadata(meta_file: Path, data: dict) -> None:
        metadata = FileQueue._load_metadata(meta_file)
        if "update_time" not in metadata:
            metadata["update_time"] = time.time()
        metadata.update(data)
        with open(meta_file, "w") as fp:
            json.dump(metadata, fp)

    @staticmethod
    def get_input_files(batch_size: Optional[int] = None) -> List[Path]:
        input_files = sorted(
            [f for f in FileQueue.STAGED_PATH.glob("*") if f.is_file() and f.suffix != ".json"],
            key=lambda f: f.stat().st_mtime,
        )
        if input_files:
            if batch_size:
                return input_files[:batch_size]
            return input_files
        return []

    @staticmethod
    def get_input_file() -> Optional[Path]:
        try:
            return next(FileQueue.get_input_files(batch_size=1))
        except StopIteration:
            return None

    @staticmethod
    def get_queued_files() -> List[Path]:
        return list(FileQueue.SOURCE_PATH.iterdir())

    @staticmethod
    def get_queue_size(self) -> int:
        return len(FileQueue.get_queued_files())
