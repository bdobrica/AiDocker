#!/usr/bin/env python3
import os
import traceback
from typing import Any, Dict

import fasttext
from daemon import AiZeroDaemon as Daemon

from .aiqueryinput import AiQueryInput


class AiQueryDaemon(Daemon):
    def worker_load(self) -> None:
        # Initialize
        MODEL_PATH = os.getenv("MODEL_PATH", "/opt/app/fasttext/lid.176.bin")
        self.model = fasttext.load_model(MODEL_PATH)

    def ai(self, input: AiQueryInput) -> Dict[str, Any]:
        try:
            model_input = input.prepare()
            model_output = self.model.predict(model_input, k=5)
            results = {k.split("_")[-1].upper(): v for k, v in zip(model_output[0], model_output[1])}
            return {"results": results}

        except Exception as e:
            if os.getenv("DEBUG", "false").lower() in ("true", "1", "on"):
                print(traceback.format_exc())
