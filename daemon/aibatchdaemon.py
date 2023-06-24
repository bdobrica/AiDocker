import json
import os
import signal
import time
from pathlib import Path
from typing import Type

from .aibatch import AiBatch
from .daemon import Daemon

__version__ = "0.8.12"


class AiBatchDaemon(Daemon):
    def __init__(self, batch_type: Type[AiBatch], *args, **kwargs):
        self.batch_type = batch_type
        super().__init__(*args, **kwargs)


    def ai(self, batch: AiBatch):
        raise NotADirectoryError("You must implement ai(batch: AiBatch)")

    def queue(self):
        STAGED_PATH = os.environ.get("STAGED_PATH", "/tmp/ai/staged")
        BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))

        staged_files = sorted(
            [
                f
                for f in Path(STAGED_PATH).glob("*")
                if f.is_file() and f.suffix != ".json"
            ],
            key=lambda f: f.stat().st_mtime,
        )

        while staged_files:
            batch = self.batch_type(staged_files=staged_files[:BATCH_SIZE])
            staged_files = staged_files[BATCH_SIZE:]
            self.ai(batch)

    def run(self):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        while True:
            self.queue()
            time.sleep(float(os.environ.get("QUEUE_LATENCY", 1.0)))
