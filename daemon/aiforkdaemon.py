import json
import os
import signal
import time
from pathlib import Path
from typing import Type

from .aiinput import AiInput
from .daemon import Daemon

__version__ = "0.8.12"


class AiForkDaemon(Daemon):
    def __init__(self, input_type: Type[AiInput], *args, **kwargs):
        self.input_type = input_type
        super().__init__(*args, **kwargs)

    def ai(self, input: AiInput):
        raise NotImplementedError(
            "You must implement ai(source_file: Path, prepared_file: Path, meta_file: Path)"
        )

    def queue(self):
        STAGED_PATH = os.environ.get("STAGED_PATH", "/tmp/ai/staged")
        MAX_FORK = int(os.environ.get("MAX_FORK", 8))

        staged_files = sorted(
            [
                f
                for f in Path(STAGED_PATH).glob("*")
                if f.is_file() and f.suffix != ".json"
            ],
            key=lambda f: f.stat().st_mtime,
        )

        while self.input_type.get_queue_size() < MAX_FORK and staged_files:
            input = self.input_type(staged_files.pop(0))
            self.ai(input)

    def run(self):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        while True:
            self.queue()
            time.sleep(float(os.environ.get("QUEUE_LATENCY", 1.0)))
