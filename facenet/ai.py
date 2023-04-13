#!/usr/bin/env python3
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from daemon import AiBatch as Batch
from daemon import AiBatchDaemon as Daemon

__version__ = "0.8.12"


class AiBatch(Batch):
    ANGLES = {
        None: "normal",
        cv2.ROTATE_180: "180",
    }
    BORDER_COLOR = (114, 114, 114)

    @staticmethod
    def load_image(file_path: Path, angle: Optional[int] = None) -> np.ndarray:
        img = cv2.imread(file_path.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if angle is not None:
            img = cv2.rotate(img, angle)
        return img

    @staticmethod
    def prepare_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
        if image.shape[0] == height and image.shape[1] == width:
            return image
        return cv2.copyMakeBorder(
            image,
            0,
            height - image.shape[0],
            0,
            width - image.shape[1],
            cv2.BORDER_CONSTANT,
            value=AiBatch.BORDER_COLOR,
        )

    @staticmethod
    def flatten_list(list_of_lists: List[List]) -> List:
        return [item for sublist in list_of_lists for item in sublist]

    @staticmethod
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

    @staticmethod
    def rotate_box(box: tuple, im_w: int, im_h: int, angle: int):
        x1, y1, x2, y2 = box
        x1, y1 = AiBatch.rotate_p(x1, y1, im_w, im_h, angle)
        x2, y2 = AiBatch.rotate_p(x2, y2, im_w, im_h, angle)
        return (
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        )

    @staticmethod
    def intersect_box(box_A: tuple, box_B: tuple):
        xa, ya, xb, yb = box_A
        xc, yc, xd, yd = box_B
        xA = max(xa, xc)
        yA = max(ya, yc)
        xB = min(xb, xd)
        yB = min(yb, yd)
        return abs(max(0, xB - xA) * max(0, yB - yA))

    def prepare(self) -> np.ndarray:
        # Load the images
        images = [
            AiBatch.load_image(source_file, angle)
            for source_file in self.source_files
            for angle in AiBatch.ANGLES
        ]

        # Compute maximum image size
        self.shapes = [(image.shape[0], image.shape[1]) for image in images]
        self.im_h = max([shape[0] for shape in self.shapes])
        self.im_w = max([shape[1] for shape in self.shapes])

        # Augment images
        images = [
            AiBatch.prepare_image(image, self.im_w, self.im_h)
            for image in images
        ]

        # Build batch input
        return np.stack([metadata["image"] for metadata in self.metadata])

    def serve(self, inference_data: Tuple[np.ndarray, np.ndarray]) -> None:
        for file_idx in range(len(self.metadata)):
            faces = []
            for angle_idx, angle in enumerate(AiBatch.ANGLES):
                orientation = AiBatch.ANGLES[angle]
                image_num = file_idx * len(AiBatch.ANGLES) + angle_idx
                shape = self.shapes[image_num]
                detections = inference_data[0][image_num]
                confidences = inference_data[0][image_num]
                # Skip no detections
                if detections is None:
                    continue

                faces_ = []
                for box_idx, box in enumerate(detections):
                    box = box.astype(int)
                    confidence = float(confidences[box_idx])
                    # Skip invalid detections
                    if (
                        box[::2].max() > self.im_h
                        or box[1::2].max() > self.im_w
                    ):
                        continue
                    # Rotate detection
                    box = AiBatch.rotate_box(box, shape[1], shape[0], angle)

                    # Find intersection with other detections
                    swap = None
                    area = 0
                    for face_idx, face in enumerate(faces):
                        int_area = AiBatch.intersect_box(box, face["box"])
                        if int_area > area:
                            swap = face_idx
                            area = int_area

                    if swap is not None:
                        if confidence > faces[swap]["conf"]:
                            faces[swap] = {
                                "box": box,
                                "conf": confidence,
                                "orientation": orientation,
                            }
                    else:
                        faces_.append(
                            {
                                "box": box,
                                "conf": confidence,
                                "orientation": orientation,
                            }
                        )
                faces.extend(faces_)

            results = [
                {
                    "x": float(0.5 * (face["box"][0] + face["box"][2])),
                    "y": float(0.5 * (face["box"][1] + face["box"][3])),
                    "w": float(abs(face["box"][0] - face["box"][2])),
                    "h": float(abs(face["box"][1] - face["box"][3])),
                    "area": float(
                        abs(face["box"][0] - face["box"][2])
                        * abs(face["box"][1] - face["box"][3])
                    ),
                    "conf": face["conf"],
                    "orientation": face["orientation"],
                }
                for face in faces
            ]

            # Save results
            with self.prepared_files[file_idx].open("w") as fp:
                json.dump(results, fp)


class AiDaemon(Daemon):
    def load(self) -> None:
        # Initialize
        MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cpu")
        if MODEL_DEVICE == "cuda" and not torch.cuda.is_available():
            MODEL_DEVICE = "cpu"
        self.device = torch.device(MODEL_DEVICE)

        # The MTCNN model has the weights inside facenet_pytorch
        self.model = MTCNN(keep_all=True, device=self.device)

    def ai(self, batch: AiBatch) -> None:
        try:
            model_input = batch.prepare()
            model_output = self.model.detect(model_input)
            batch.serve(model_output)
            batch.update_metadata({"processed", "true"})

        except Exception as e:
            batch.update_metadata({"processed": "error"})
            if os.environ.get("DEBUG", "false").lower() in ("true", "1", "on"):
                print(traceback.format_exc())


if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AiDaemon(batch_type=AiBatch, pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
