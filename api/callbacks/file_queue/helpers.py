"""
File-queue helper functions.
"""
import os
from pathlib import Path
from typing import List


def get_metadata_path(file_token: str) -> Path:
    """
    Given a file token, return the path to the metadata file. The metadata file is unique to the file token and does not
    change directory when the file that it references is passed through the file queue.
    :param file_token: the file token, an API_FILE_HASHER of the original file
    :return: the path to the metadata file
    """
    STAGED_PATH = Path(os.getenv("STAGED_PATH", "/tmp/ai/staged"))
    return STAGED_PATH / (file_token + ".json")


def get_staged_path(file_token: str, file_suffix: str) -> Path:
    """
    Given a file token and a file suffix, return the path to the staged file.
    :param file_token: the file token, an API_FILE_HASHER of the original file
    :param file_suffix: the file suffix, the leading period is optional, case-insensitive
    :return: the path to the staged file
    """
    STAGED_PATH = Path(os.getenv("STAGED_PATH", "/tmp/ai/staged"))
    return STAGED_PATH / (file_token + "." + file_suffix.strip(".").lower())


def get_prepared_path(file_token: str, file_suffix: str = "json") -> Path:
    """
    Given a file token and a file suffix, return the path to the prepared file.
    Most of the time, you have only one prepared file, so this function is a fast way to get the path to that file. If
    you have multiple prepared files, use get_prepared_paths() instead.
    :param file_token: the file token, an API_FILE_HASHER of the original file
    :param file_suffix: the file suffix, the leading period is optional, case-insensitive
    :return: the path to the prepared file
    """
    PREPARED_PATH = Path(os.getenv("PREPARED_PATH", "/tmp/ai/prepared"))
    return PREPARED_PATH / (file_token + "." + file_suffix.strip(".").lower())


def get_staged_paths(file_token: str) -> List[Path]:
    """
    Given a file token, return the paths to the staged files (there may be more than one).
    :param file_token: the file token, an API_FILE_HASHER of the original file
    :return: the paths to the staged files
    """
    STAGED_PATH = Path(os.getenv("STAGED_PATH", "/tmp/ai/staged"))
    return [path for path in STAGED_PATH.glob(file_token + "*") if path.is_file() and path.suffix != ".json"]


def get_source_paths(file_token: str) -> List[Path]:
    """
    Given a file token, return the paths to the source files (there may be more than one).
    :param file_token: the file token, an API_FILE_HASHER of the original file
    :return: the paths to the source files
    """
    SOURCE_PATH = Path(os.getenv("SOURCE_PATH", "/tmp/ai/source"))
    return [path for path in SOURCE_PATH.glob(file_token + "*") if path.is_file() and path.suffix != ".json"]


def get_prepared_paths(file_token: str) -> List[Path]:
    """
    Given a file token, return the paths to the prepared files (there may be more than one).
    :param file_token: the file token, an API_FILE_HASHER of the original file
    :return: the paths to the prepared files
    """
    PREPARED_PATH = Path(os.getenv("PREPARED_PATH", "/tmp/ai/prepared"))
    return [path for path in PREPARED_PATH.glob(file_token + "*")]


def clean_files(file_token: str) -> None:
    """
    Given a file token, delete all files associated with that file token. Metadata files are last to be deleted. Cleanup
    is done ignoring errors.
    :param file_token: the file token, an API_FILE_HASHER of the original file
    """
    paths = get_staged_paths(file_token) + get_source_paths(file_token) + get_prepared_paths(file_token)
    for path in paths:
        if path.exists():
            path.unlink()
    path = get_metadata_path(file_token)
    if path.exists():
        path.unlink()
