#!/usr/bin/env python3
import os
import traceback
from pathlib import Path

import pandas as pd

import ppscore
from daemon import AiForkDaemon as Daemon
from daemon import AiInput as Input

__version__ = "0.8.13"


class AiInput(Input):
    DEFAULT_EXTENSION = "csv"

    def prepare(self) -> pd.DataFrame:
        return pd.read_csv(self.source_file)

    def serve(self, inference_data: pd.DataFrame) -> None:
        inference_data.to_csv(self.prepared_file, index=False, header=True)


class AiDaemon(Daemon):
    def load(self) -> None:
        pass

    def ai(self, input: AiInput) -> str:
        try:
            model_input = input.prepare()
            model_output = ppscore.matrix(model_input)
            input.serve(model_output)
            input.update_metadata({"processed": "true"})

        except Exception as e:
            input.update_metadata({"processed": "error"})
            if os.getenv("DEBUG", "false").lower() in ("true", "1", "on"):
                print(traceback.format_exc())


if __name__ == "__main__":
    CHROOT_PATH = os.getenv("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.getenv("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AiDaemon(input_type=AiInput, pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
