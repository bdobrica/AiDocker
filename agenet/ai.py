#!/usr/bin/env python3
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import cv2
import dlib
import numpy as np
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from daemon import Daemon

__version__ = "0.8.12"


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
            MODEL_PATH = os.environ.get(
                "MODEL_PATH", "/opt/app/EfficientNetB3_224_weights.11-3.44.hdf5"
            )
            EFFICIENTNETB3_PATH = os.environ.get(
                "EFFICIENTNETB3_PATH", "/opt/app/efficientnetb3_notop.h5"
            )
            IMAGE_SIZE = 224
            MARGIN = 0.4

            # Initialize
            base_model = applications.EfficientNetB3(
                include_top=False,
                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                pooling="avg",
                weights=EFFICIENTNETB3_PATH,
            )
            features = base_model.output
            pred_gender = Dense(
                units=2, activation="softmax", name="pred_gender"
            )(features)
            pred_age = Dense(units=101, activation="softmax", name="pred_age")(
                features
            )
            model = Model(
                inputs=base_model.input, outputs=[pred_gender, pred_age]
            )
            detector = dlib.get_frontal_face_detector()

            # Load model
            model.load_weights(MODEL_PATH)

            # Load image
            img_orig = cv2.imread(str(source_file))
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = img_orig.shape

            detected = detector(img_orig, 1)
            faces = np.empty((len(detected), IMAGE_SIZE, IMAGE_SIZE, 3))

            results = []
            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = (
                        d.left(),
                        d.top(),
                        d.right() + 1,
                        d.bottom() + 1,
                        d.width(),
                        d.height(),
                    )
                    xw1 = max(int(x1 - MARGIN * w), 0)
                    yw1 = max(int(y1 - MARGIN * h), 0)
                    xw2 = min(int(x2 + MARGIN * w), img_w - 1)
                    yw2 = min(int(y2 + MARGIN * h), img_h - 1)
                    faces[i] = cv2.resize(
                        img_orig[yw1 : yw2 + 1, xw1 : xw2 + 1],
                        (IMAGE_SIZE, IMAGE_SIZE),
                    )
                    results.append(
                        {
                            "x": 0.5 * (x1 + x2),
                            "y": 0.5 * (y1 + y2),
                            "w": w,
                            "h": h,
                        }
                    )

                prediction = model.predict(faces)
                predicted_genders = prediction[0].astype(float)
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = prediction[1].astype(float).dot(ages).flatten()

                for i, result in enumerate(results):
                    result["female"] = predicted_genders[i, 0]
                    result["male"] = predicted_genders[i, 1]
                    result["age"] = predicted_ages[i]

            json_file = prepared_file.with_suffix(".json")
            with json_file.open("w") as f:
                json.dump({"results": results}, f)

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
