#!/usr/bin/env python3
import json
import os
import signal
import time
from pathlib import Path
from typing import List

from daemon import Daemon

__version__ = "0.8.0"


def get_metadata_path(file_token: str) -> Path:
    STAGED_PATH = Path(os.getenv("STAGED_PATH", "/tmp/ai/staged"))
    return STAGED_PATH / (file_token + ".json")


def get_metadata_paths() -> List[Path]:
    STAGED_PATH = Path(os.getenv("STAGED_PATH", "/tmp/ai/staged"))
    return [path for path in STAGED_PATH.glob("*.json") if path.is_file()]


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


class Cleaner(Daemon):
    def clean(self):
        FILE_LIFETIME = os.getenv("API_CLEANER_FILE_LIFETIME", "1800.0")
        lifetime = 1.1 * float(FILE_LIFETIME)

        for meta_file in get_metadata_paths():
            with meta_file.open("r") as fp:
                try:
                    file_metadata = json.load(fp)
                except:
                    file_metadata = {}

            if float(file_metadata.get("upload_time", 0)) + lifetime < time.time():
                file_token = meta_file.stem
                clean_files(file_token)

    def run(self):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        FILE_LIFETIME = os.getenv("API_CLEANER_FILE_LIFETIME", "1800.0")
        sleep_interval = min(
            0.05 * float(FILE_LIFETIME),
            float(os.getenv("API_CLEANER_INTERVAL", "5.0")),
        )
        while True:
            self.clean()
            time.sleep(sleep_interval)


if __name__ == "__main__":
    CHROOT_PATH = os.getenv("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.getenv("PIDFILE_PATH", "/opt/app/run/cleaner.pid")

    Cleaner(pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
