import os
from pathlib import Path
from typing import List


def get_metadata_path(file_token: str) -> Path:
    STAGED_PATH = Path(os.getenv("STAGED_PATH", "/tmp/ai/staged"))
    return STAGED_PATH / (file_token + ".json")


def get_staged_path(file_token: str, file_suffix: str) -> Path:
    STAGED_PATH = Path(os.getenv("STAGED_PATH", "/tmp/ai/staged"))
    return STAGED_PATH / (file_token + "." + file_suffix.strip(".").lower())


def get_json_path(file_token: str, file_suffix: str = "json") -> Path:
    PREPARED_PATH = Path(os.getenv("PREPARED_PATH", "/tmp/ai/prepared"))
    return PREPARED_PATH / (file_token + "." + file_suffix.strip(".").lower())


def get_staged_paths(file_token: str) -> List[Path]:
    STAGED_PATH = Path(os.getenv("STAGED_PATH", "/tmp/ai/staged"))
    return [path for path in STAGED_PATH.glob(file_token + "*") if path.is_file() and path.suffix != ".json"]


def get_source_paths(file_token: str) -> List[Path]:
    SOURCE_PATH = Path(os.getenv("SOURCE_PATH", "/tmp/ai/source"))
    return [path for path in SOURCE_PATH.glob(file_token + "*") if path.is_file() and path.suffix != ".json"]


def get_prepared_paths(file_token: str) -> List[Path]:
    PREPARED_PATH = Path(os.getenv("PREPARED_PATH", "/tmp/ai/prepared"))
    return [path for path in PREPARED_PATH.glob(file_token + "*")]


def clean_files(file_token: str) -> None:
    paths = get_staged_paths(file_token) + get_source_paths(file_token) + get_prepared_paths(file_token)
    for path in paths:
        if path.exists():
            path.unlink()
    path = get_metadata_path(file_token)
    if path.exists():
        path.unlink()
