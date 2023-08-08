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
