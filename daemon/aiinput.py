import json
import os
import time
from pathlib import Path


class AiInput:
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
        metadata = AiInput._load_metadata(meta_file)
        if "update_time" not in metadata:
            metadata["update_time"] = time.time()
        metadata.update(data)
        with open(meta_file, "w") as fp:
            json.dump(metadata, fp)

    @staticmethod
    def get_queued_files() -> list:
        return list(AiInput.SOURCE_PATH.iterdir())

    @staticmethod
    def get_queue_size(self) -> int:
        return len(AiInput.get_queued_files())

    def __init__(self, staged_file: Path):
        self.staged_file = staged_file
        self.meta_file = self.staged_file.with_suffix(".json")
        self.source_file = self.SOURCE_PATH / self.staged_file.name
        self.metadata = self._load_metadata(self.meta_file)
        self.prepared_file = self.PREPARED_PATH / (
            self.staged_file.stem + "." + self.metadata.get("output_extension", self.DEFAULT_EXTENSION)
        )
        self.staged_file.rename(self.source_file)

    def prepare(self) -> any:
        raise NotImplementedError("You must implement prepare() -> any")

    def serve(self, inference_data: any) -> None:
        raise NotImplementedError("You must implement serve(inference_data: any)")

    def update_metadata(self, data: dict) -> None:
        AiInput._update_metadata(self.meta_file, data)

    def __del__(self):
        self.source_file.unlink()
