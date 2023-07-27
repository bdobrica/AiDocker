#!/usr/bin/env python3
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from daemon import Daemon
from nudenet import NudeDetector

__version__ = "0.8.13"


class AIDaemon(Daemon):
    def load_metadata(self, meta_file: Path) -> dict:
        if not meta_file.is_file():
            return {}
        with open(meta_file, "r") as fp:
            return json.load(fp)

    def update_metadata(self, meta_file: Path, data: dict) -> None:
        metadata = self.load_metadata(meta_file)
        if "update_time" not in metadata:
            metadata["update_time"] = time.time()
        metadata.update(data)
        with open(meta_file, "w") as fp:
            json.dump(metadata, fp)

    def ai(self, source_file: Path, prepared_file: Path, meta_file: Path) -> None:
        pid = os.fork()
        if pid != 0:
            return

        try:
            # Load model
            model = NudeDetector()
            img_orig = cv2.imread(str(source_file))

            # Load metadata
            metadata = self.load_metadata(meta_file)

            # Inference
            kwargs = {}
            if metadata.get("fast", "no").lower() == "yes":
                kwargs["fast"] = True
            results = model.detect(str(source_file), **kwargs)
            results = filter(
                lambda item: item["score"] > os.getenv("API_THRESHOLD", 0.5),
                results,
            )

            API_NUDENET_KEEP_LABELS = os.getenv("API_NUDENET_KEEP_LABELS", "")
            if API_NUDENET_KEEP_LABELS:
                results = filter(
                    lambda item: item["label"] in API_NUDENET_KEEP_LABELS.split(","),
                    results,
                )
            API_NUDENET_DROP_LABELS = os.getenv("API_NUDENET_DROP_LABELS", "")
            if API_NUDENET_DROP_LABELS:
                results = filter(
                    lambda item: item["label"] not in API_NUDENET_DROP_LABELS.split(","),
                    results,
                )

            # Do censoring
            img_copy = img_orig.copy()
            if metadata.get("censor", "no").lower() == "yes":
                API_NUDENET_CENSOR_TYPE = os.getenv("API_NUDENET_CENSOR_TYPE", "blackbox")
                if results:
                    for item in results:
                        box = tuple(item["box"])
                        if API_NUDENET_CENSOR_TYPE == "blackbox":
                            img_copy = cv2.rectangle(img_copy, box[:2], box[2:], (0, 0, 0), -1)
                        elif API_NUDENET_CENSOR_TYPE == "blur":
                            img_box = img_copy[box[1] : box[3], box[0] : box[2], :]
                            box_height, box_width = img_box.shape[:2]
                            box_blur = (
                                1 + 2 * (box_height // 2),
                                1 + 2 * (box_width // 2),
                            )
                            img_box = cv2.GaussianBlur(img_box, box_blur, cv2.BORDER_DEFAULT)
                            img_copy[box[1] : box[3], box[0] : box[2], :] = img_box

                cv2.imwrite(str(prepared_file), img_copy)
            else:
                json_file = prepared_file.with_suffix(".json")
                with json_file.open("w") as f:
                    json.dump({"results": list(results)}, f)
            self.update_metadata(
                meta_file,
                {
                    "processed": "true",
                },
            )
        except Exception as e:
            if os.getenv("DEBUG", "false").lower() in ("true", "1", "on"):
                print(traceback.format_exc())
            self.update_metadata(
                meta_file,
                {
                    "processed": "error",
                },
            )

        source_file.unlink()
        sys.exit()

    def queue(self) -> None:
        STAGED_PATH = os.getenv("STAGED_PATH", "/tmp/ai/staged")
        SOURCE_PATH = os.getenv("SOURCE_PATH", "/tmp/ai/source")
        PREPARED_PATH = os.getenv("PREPARED_PATH", "/tmp/ai/prepared")
        MAX_FORK = int(os.getenv("MAX_FORK", 8))
        CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 4096))

        staged_files = sorted(
            [f for f in Path(STAGED_PATH).glob("*") if f.is_file() and f.suffix != ".json"],
            key=lambda f: f.stat().st_mtime,
        )
        source_files = [f for f in Path(SOURCE_PATH).glob("*") if f.is_file()]
        source_files_count = len(source_files)

        while source_files_count < MAX_FORK and staged_files:
            source_files_count += 1
            staged_file = staged_files.pop(0)

            meta_file = staged_file.with_suffix(".json")
            source_file = Path(SOURCE_PATH) / staged_file.name
            prepared_file = Path(PREPARED_PATH) / (staged_file.stem + staged_file.suffix)

            with staged_file.open("rb") as src_fp, source_file.open("wb") as dst_fp:
                while True:
                    chunk = src_fp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    dst_fp.write(chunk)

            staged_file.unlink()
            self.ai(source_file, prepared_file, meta_file)

    def run(self) -> None:
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        while True:
            self.queue()
            time.sleep(float(os.getenv("QUEUE_LATENCY", 1.0)))


if __name__ == "__main__":
    CHROOT_PATH = os.getenv("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.getenv("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AIDaemon(pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
