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
from torch.autograd import Variable
from torchvision import transforms

from daemon import AiForkDaemon as Daemon
from daemon import AiInput as Input
from u2net import U2NET

__version__ = "0.8.12"


class AiInput(Input):
    @staticmethod
    def _rescale_t(image: np.ndarray, output_size: int = 320) -> np.ndarray:
        return cv2.resize(
            image,
            dsize=(output_size, output_size),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _to_tensor_lab(image: np.ndarray, flag: int = 0) -> torch.Tensor:
        image_ = image / np.max(image)

        image_[:, :, 0] = (image_[:, :, 0] - 0.485) / 0.229
        image_[:, :, 1] = (image_[:, :, 1] - 0.456) / 0.224
        image_[:, :, 2] = (image_[:, :, 2] - 0.406) / 0.225

        image_ = image_.transpose((2, 0, 1))
        image_ = image_[np.newaxis, :, :, :]

        return torch.from_numpy(image_)

    def prepare(self) -> Variable:
        img = cv2.imread(self.source_file.as_posix())
        self.shape = img.shape
        self.out_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        img = img[:, :, ::-1]
        img_t = Variable(
            transforms.Compose([AiInput._rescale_t, AiInput._to_tensor_lab])(
                img
            ).type(torch.FloatTensor)
        )
        return img_t

    def serve(self, inference_data: torch.Tensor) -> None:
        pred = inference_data[0][:, 0, :, :]
        ma = torch.max(pred)
        mi = torch.min(pred)
        pred = (pred - mi) / (ma - mi)
        pred_np = pred.squeeze().cpu().data.numpy()

        sal_map = (pred_np * 255).astype("uint8")
        sal_map = cv2.resize(
            sal_map, self.shape[1::-1], interpolation=cv2.INTER_AREA
        )
        out_im[:, :, 3] = sal_map
        out_im = out_im.astype(float)

        im_h, im_w = out_im.shape[:2]

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
                        background_im[:, :, 3].reshape(out_im.shape[:2] + (1,))
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

        if background_im is None and metadata.get("type", "") != "image/png":
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

    def ai(self, input: AiInput) -> None:
        pid = os.fork()
        if pid != 0:
            return

        try:
            net = U2NET(3, 1)
            MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/app/u2net.pth")
            net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            net.eval()

            output = net(input.prepare())
            input.serve(output)

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
