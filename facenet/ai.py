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
from facenet_pytorch import MTCNN

from daemon import Daemon

__version__ = "0.8.8"


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
            im = cv2.imread(str(source_file))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_h, im_w, _ = im.shape

            # Data engineering tools
            def rotate_p(x: float, y: float, im_w: int, im_h: int, angle: int):
                if angle == cv2.ROTATE_90_CLOCKWISE:
                    res = (y, im_h - x)
                elif angle == cv2.ROTATE_180:
                    res = (im_w - x, im_h - y)
                elif angle == cv2.ROTATE_90_COUNTERCLOCKWISE:
                    res = (im_w - y, x)
                else:
                    res = (x, y)
                return res

            def rotate_box(box: tuple, im_w: int, im_h: int, angle: int):
                x1, y1, x2, y2 = box
                x1, y1 = rotate_p(x1, y1, im_w, im_h, angle)
                x2, y2 = rotate_p(x2, y2, im_w, im_h, angle)
                return (
                    min(x1, x2),
                    min(y1, y2),
                    max(x1, x2),
                    max(y1, y2),
                )

            def intersect_box(box_A: tuple, box_B: tuple):
                xa, ya, xb, yb = box_A
                xc, yc, xd, yd = box_B
                xA = max(xa, xc)
                yA = max(ya, yc)
                xB = min(xb, xd)
                yB = min(yb, yd)
                return abs(max(0, xB - xA) * max(0, yB - yA))

            # Detect faces
            angles = {
                None: "normal",
                # cv2.ROTATE_90_CLOCKWISE: "90cw",
                cv2.ROTATE_180: "180",
                # cv2.ROTATE_90_COUNTERCLOCKWISE: "90ccw",
            }

            # Check different orientations and choose the best faces from each
            faces = []
            for angle, orientation in angles.items():
                detected, confs = model.detect(
                    im if angle is None else cv2.rotate(im, angle)
                )
                faces_ = []
                if detected is not None:
                    for i, box in enumerate(detected):
                        box = rotate_box(box, im_w, im_h, angle)
                        conf = float(confs[i])

                        swap = None
                        area = 0
                        for j, face in enumerate(faces):
                            int_area = intersect_box(box, face["box"])
                            if int_area > area:
                                swap = j
                                area = int_area

                        if swap is not None:
                            if conf > faces[swap]["conf"]:
                                print("conf", conf, "new!")
                                faces[swap] = {
                                    "box": box,
                                    "conf": conf,
                                    "orientation": orientation,
                                }
                        else:
                            faces_.append(
                                {
                                    "box": box,
                                    "conf": conf,
                                    "orientation": orientation,
                                }
                            )
                faces.extend(faces_)

            results = [
                {
                    "x": float(0.5 * (f["box"][0] + f["box"][2])),
                    "y": float(0.5 * (f["box"][1] + f["box"][3])),
                    "w": float(abs(f["box"][0] - f["box"][2])),
                    "h": float(abs(f["box"][1] - f["box"][3])),
                    "area": float(
                        abs(f["box"][0] - f["box"][2])
                        * abs(f["box"][1] - f["box"][3])
                    ),
                    "conf": f["conf"],
                    "orientation": angles.get(f["orientation"]),
                }
                for f in faces
            ]
            results = sorted(results, key=lambda x: x["area"], reverse=True)

            json_file = prepared_file.with_suffix(".json")
            with json_file.open("w") as f:
                json.dump({"results": results}, f)

        except Exception as e:
            if os.environ.get("DEBUG", "false").lower() in ("true", "1", "on"):
                print(traceback.format_exc())

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
