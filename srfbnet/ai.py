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
import torch

from daemon import Daemon
from srfbnet import GMFN

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

    def ai(
        self, source_file: Path, prepared_file: Path, meta_file: Path
    ) -> None:
        pid = os.fork()
        if pid != 0:
            return

        try:
            # Load metadata
            metadata = self.load_metadata(meta_file)
            scale = int(metadata.get("scale", 2))

            # Detect device
            MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cuda:0")
            if (
                MODEL_DEVICE.startswith("cuda")
                and not torch.cuda.is_available()
            ):
                MODEL_DEVICE = "cpu"
            self.device = torch.device(MODEL_DEVICE)

            # Load model
            model = GMFN(upscale_factor=scale)
            MODEL_PATH = os.environ.get(
                "MODEL_PATH",
                "/opt/app/gmfn_x{scale}.pth",
            )
            ckpt = torch.load(
                MODEL_PATH.format(scale=scale), map_location=self.device
            )
            model.load_state_dict(ckpt, strict=True)
            _ = model.eval()

            # Prepare image
            im = cv2.imread(str(source_file))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            if len(im.shape) == 2:
                im = im[:, :, None]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
            elif im.shape[2] == 4:
                im = im[:, :, 0:3]

            # Inference
            im = np.ascontiguousarray(im.transpose((2, 0, 1)))
            im_t = torch.from_numpy(im[np.newaxis, :, :, :]).float()
            out_im_t = model.forward(im_t)
            if isinstance(out_im_t, list):
                out_im_t = out_im_t[-1]
            out_im_t = out_im_t.data[0].float().cpu()
            out_im_t = out_im_t.clamp(0, 255).round()
            out_im = out_im_t.numpy().transpose((1, 2, 0))
            out_im = out_im[:, :, ::-1]

            cv2.imwrite(str(prepared_file), out_im.astype(np.uint8))
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
            time.sleep(float(os.environ.get("QUEUE_LATENCY", 1.0)))


if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AIDaemon(pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
