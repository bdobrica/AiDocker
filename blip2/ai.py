#!/usr/bin/env python3
import json
import os
import random
import signal
import time
from pathlib import Path

import cv2
import numpy as np

from daemon import Daemon

__version__ = "0.8.8"


class AIDaemon(Daemon):
    def load(self):
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        # Try reproducing the results
        torch.manual_seed(42)

        self.device = torch.device("cpu")

        MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/app/blip2")

        self.processor = Blip2Processor.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)

    def ai(self, source_file, prepared_file, **metadata):
        # Deep Learning models are not fork-safe (so no multiprocessing)
        try:
            # Try reproducing the results
            np.random.seed(42)
            random.seed(42)

            # Load image
            im_orig = cv2.imread(str(source_file))
            im_orig = cv2.cvtColor(im_orig, cv2.COLOR_BGR2RGB)

            # Inference
            inputs = self.processor(im_orig, return_tensors="pt")
            preds = self.model.generate(**inputs)

            results = [{"captions": self.processor.decode(pred, skip_special_tokens=True).strip()} for pred in preds]

            json_file = prepared_file.with_suffix(".json")
            with json_file.open("w") as f:
                json.dump({"results": results}, f)

        except Exception as e:
            pass

        source_file.unlink()

    def queue(self):
        STAGED_PATH = os.environ.get("STAGED_PATH", "/tmp/ai/staged")
        SOURCE_PATH = os.environ.get("SOURCE_PATH", "/tmp/ai/source")
        PREPARED_PATH = os.environ.get("PREPARED_PATH", "/tmp/ai/prepared")
        CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 4096))

        staged_files = sorted(
            [f for f in Path(STAGED_PATH).glob("*") if f.is_file() and f.suffix != ".json"],
            key=lambda f: f.stat().st_mtime,
        )
        source_files = [f for f in Path(SOURCE_PATH).glob("*") if f.is_file()]

        while not source_files and staged_files:
            staged_file = staged_files.pop(0)

            meta_file = staged_file.with_suffix(".json")
            if meta_file.is_file():
                with meta_file.open("r") as fp:
                    try:
                        image_metadata = json.load(fp)
                    except:
                        image_metadata = {}
            image_metadata = {
                **{
                    "extension": staged_file.suffix,
                    "background": "",
                },
                **image_metadata,
            }

            source_file = Path(SOURCE_PATH) / staged_file.name
            prepared_file = Path(PREPARED_PATH) / (staged_file.stem + image_metadata["extension"])

            with staged_file.open("rb") as src_fp, source_file.open("wb") as dst_fp:
                while True:
                    chunk = src_fp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    dst_fp.write(chunk)

            staged_file.unlink()
            self.ai(source_file, prepared_file, **image_metadata)

    def run(self):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        while True:
            self.queue()
            time.sleep(0.1)


if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AIDaemon(pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
