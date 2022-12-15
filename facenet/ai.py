#!/usr/bin/env python3
import json
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import torch
from facenet_pytorch import MTCNN

from daemon import Daemon

__version__ = "0.8.7"


class AIDaemon(Daemon):
    def ai(self, source_file, prepared_file, **metadata):
        pid = os.fork()
        if pid != 0:
            return

        try:
            # Initialize
            MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cpu")
            if MODEL_DEVICE == "cuda" and not torch.cuda.is_available():
                MODEL_DEVICE = "cpu"
            self.device = torch.device(MODEL_DEVICE)

            # The MTCNN model has the weights inside facenet_pytorch
            model = MTCNN(keep_all=True, device=self.device)

            # Load image
            img_orig = cv2.imread(str(source_file))
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

            detected, confs = model.detect(img_orig)

            results = []
            if len(detected) > 0:
                for i, box in enumerate(detected):
                    x1, y1, x2, y2 = box.tolist()

                    results.append(
                        {
                            "x": 0.5 * (x1 + x2),
                            "y": 0.5 * (y1 + y2),
                            "w": abs(x2 - x1),
                            "h": abs(y2 - y1),
                            "conf": confs[i],
                        }
                    )

            json_file = prepared_file.with_suffix(".json")
            with json_file.open("w") as f:
                json.dump({"results": results}, f)

        except Exception as e:
            pass

        source_file.unlink()
        sys.exit()

    def queue(self):
        STAGED_PATH = os.environ.get("STAGED_PATH", "/tmp/ai/staged")
        SOURCE_PATH = os.environ.get("SOURCE_PATH", "/tmp/ai/source")
        PREPARED_PATH = os.environ.get("PREPARED_PATH", "/tmp/ai/prepared")
        MAX_FORK = int(os.environ.get("MAX_FORK", 8))
        CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 4096))

        staged_files = sorted(
            [
                f
                for f in Path(STAGED_PATH).glob("*")
                if f.is_file() and f.suffix != ".json"
            ],
            key=lambda f: f.stat().st_mtime,
        )
        source_files = [f for f in Path(SOURCE_PATH).glob("*") if f.is_file()]
        source_files_count = len(source_files)

        while source_files_count < MAX_FORK and staged_files:
            source_files_count += 1
            staged_file = staged_files.pop(0)

            meta_file = staged_file.with_suffix(".json")
            if meta_file.is_file():
                with meta_file.open("r") as fp:
                    try:
                        image_metadata = json.load(fp)
                    except:
                        image_metadata = {}

            source_file = Path(SOURCE_PATH) / staged_file.name
            prepared_file = Path(PREPARED_PATH) / (
                staged_file.stem + image_metadata["extension"]
            )

            with staged_file.open("rb") as src_fp, source_file.open(
                "wb"
            ) as dst_fp:
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
            time.sleep(1.0)


if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AIDaemon(pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
