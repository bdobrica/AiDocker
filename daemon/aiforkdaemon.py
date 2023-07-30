import os
import signal
import sys
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

    def ai(self, input: AiInput) -> None:
        raise NotImplementedError("You must implement ai(input: AiInput) -> None")

    def queue(self):
        STAGED_PATH = os.getenv("STAGED_PATH", "/tmp/ai/staged")
        MAX_FORK = int(os.getenv("MAX_FORK", 8))

        staged_files = sorted(
            [f for f in Path(STAGED_PATH).glob("*") if f.is_file() and f.suffix != ".json"],
            key=lambda f: f.stat().st_mtime,
        )

        while self.input_type.get_queue_size() < MAX_FORK and staged_files:
            staged_file = staged_files.pop(0)
            pid = os.fork()
            if pid == 0:
                input = self.input_type(staged_file)
                self.ai(input)
                sys.exit(0)

    def run(self):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        while True:
            self.queue()
            time.sleep(float(os.getenv("QUEUE_LATENCY", 1.0)))
