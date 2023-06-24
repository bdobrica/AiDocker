#!/usr/bin/env python3
import os
import traceback
from pathlib import Path
from typing import Tuple

import pandas as pd

from daemon import AiForkDaemon as Daemon
from daemon import AiInput as Input
from prophet import Prophet

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

            history = model_input[model_input["y"].notnull()]
            future = model_input[model_input["y"].isnull()]

            model = Prophet()
            model.fit(history)
            forecast = model.predict(future)

            model_output = pd.concat([history, forecast], ignore_index=True)

            input.serve(model_output)
            input.update_metadata({"processed": "true"})

        except Exception as e:
            input.update_metadata({"processed": "error"})
            if os.environ.get("DEBUG", "false").lower() in ("true", "1", "on"):
                print(traceback.format_exc())


if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AiDaemon(input_type=AiInput, pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
