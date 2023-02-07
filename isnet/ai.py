#!/usr/bin/env python3
import json
import os
import re
import signal
import sys
import time
import traceback
from pathlib import Path
from urllib import request

import cv2
import numpy as np
import torch

from daemon import Daemon
from isnet import ISNetDIS
from isnet.utils import inference

__version__ = "0.8.11"


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

    def ai(
        self, source_file: Path, prepared_file: Path, meta_file: Path
    ) -> None:
        pid = os.fork()
        if pid != 0:
            return

        try:
            # Load model
            model = ISNetDIS()
            MODEL_PATH = os.environ.get(
                "MODEL_PATH",
                "/opt/app/isnet-general-use.pth",
            )
            ckpt = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
            model.load_state_dict(ckpt, strict=True)
            _ = model.eval()

            im = cv2.imread(str(source_file))
            out_im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            if len(im.shape) == 2:
                im = im[:, :, None]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
            elif im.shape[2] == 4:
                im = im[:, :, 0:3]

            im_h, im_w, _ = im.shape

            # Inference
            mask = inference(model, im)

            out_im[:, :, 3] = mask
            out_im = out_im.astype(float)

            # Load metadata
            metadata = self.load_metadata(meta_file)

            background = metadata.get("background", "").strip(" #")
            if len(background) == 6:
                background = background.lower() + "ff"
            color_re = re.compile(r"(^[A-Za-z0-9]{6}$)|(^[A-Za-z0-9]{8}$)")
            background_alpha = 0
            background_im = None
            if background.startswith("http://") or background.startswith(
                "https://"
            ):
                try:
                    req = request.urlopen(background)
                    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                    background_im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                    background_im = cv2.resize(
                        background_im,
                        dsize=(im_w, im_h),
                        interpolation=cv2.INTER_AREA,
                    )
                    if background_im.shape[2] == 4:
                        background_im = background_im.astype(float)
                        background_im = background_im[:, :, :3] * np.repeat(
                            background_im[:, :, 3].reshape(
                                out_im.shape[:2] + (1,)
                            )
                            / 255.0,
                            3,
                            axis=2,
                        )
                    background_alpha = 1.0
                except:
                    background_im = None
            elif color_re.match(background):
                red = int(background[0:2], 16)
                green = int(background[2:4], 16)
                blue = int(background[4:6], 16)
                if len(background) == 8:
                    background_alpha = int(background[6:8], 16) / 255.0
                else:
                    background_alpha = 1.0
                background_im = np.full(
                    (im_h, im_w, 3), [blue, green, red], dtype=np.float32
                )

            alpha_mask = np.repeat(
                out_im[:, :, 3].reshape(out_im.shape[:2] + (1,)) / 255.0,
                3,
                axis=2,
            )
            out_im[:, :, :3] = out_im[:, :, :3] * alpha_mask

            if (
                background_im is None
                and metadata.get("type", "") != "image/png"
            ):
                background_im = np.full(
                    (im_h, im_w, 3), [255.0, 255.0, 255.0], dtype=np.float32
                )
                background_alpha = 1.0

            if background_im is not None:
                out_im[:, :, :3] = out_im[:, :, :3] + background_im[
                    :, :, :3
                ] * background_alpha * (1.0 - alpha_mask)
                out_im[:, :, 3] = 255.0

            cv2.imwrite(str(prepared_file), out_im.astype("uint8"))
            self.update_metadata(
                meta_file,
                {
                    "processed": "true",
                },
            )
        except Exception as e:
            if os.environ.get("DEBUG", "false").lower() in ("true", "1", "on"):
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
            source_file = Path(SOURCE_PATH) / staged_file.name
            prepared_file = Path(PREPARED_PATH) / (
                staged_file.stem + staged_file.suffix
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
            self.ai(source_file, prepared_file, meta_file)

    def run(self) -> None:
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        while True:
            self.queue()
            time.sleep(1.0)


if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AIDaemon(pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
