#!/usr/bin/env python3
import json
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
)

from daemon import Daemon

__version__ = "0.8.5"


class AIDaemon(Daemon):
    def ai(self, source_file, prepared_file, **metadata):
        pid = os.fork()
        if pid != 0:
            return

        try:
            MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/app/vitgpt2")

            # Initialize & Load Model
            model = VisionEncoderDecoderModel.from_pretrained(
                MODEL_PATH, local_files_only=True
            )
            feature_extractor = ViTFeatureExtractor.from_pretrained(
                MODEL_PATH, local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH, local_files_only=True
            )

            device = torch.device("cpu")
            model = model.to(device)

            # Parameters
            max_length = metadata.get("max_length", 16)
            num_beams = metadata.get("num_beams", 4)

            # Load image
            im_orig = cv2.imread(str(source_file))
            im_orig = cv2.cvtColor(im_orig, cv2.COLOR_BGR2RGB)

            # Inference
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
            pixel_values = feature_extractor(
                images=im_orig[np.newaxis, :, :, :], return_tensors="pt"
            ).pixel_values
            pixel_values = pixel_values.to(device)
            output_ids = model.generate(pixel_values, **gen_kwargs)
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            results = [{"caption": pred.strip()} for pred in preds]

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
            image_metadata = {
                **{
                    "extension": staged_file.suffix,
                    "background": "",
                },
                **image_metadata,
            }

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
