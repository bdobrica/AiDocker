#!/usr/bin/env python3
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from daemon import AiForkDaemon as Daemon
from daemon import AiInput as Input
from yolov4 import Darknet, non_max_suppression, scale_coords

__version__ = "0.8.13"


class AiInput(Input):
    IMAGE_SIZE = 640
    AUTO_SIZE = 64
    BORDER_COLOR = (114, 114, 114)
    CLASSES_PATH = os.environ.get("CLASSES_PATH", "/opt/app/coco.names")

    def _load_classes(self) -> List[str]:
        # Get names and colors
        with Path(self.CLASSES_PATH).open("r") as f:
            names = list(
                filter(None, f.read().split("\n"))
            )  # filter removes empty strings (such as last line)
        return names

    def _load_image(self) -> np.ndarray:
        self.img_orig = cv2.imread(self.source_file.as_posix())
        self.shape = self.img_orig.shape

        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        # current shape [height, width]
        shape = self.shape[:2]
        # desired new shape [height, width]
        new_shape = (self.IMAGE_SIZE, self.IMAGE_SIZE)
        # Scale ratio (new / old)
        ratio_ = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Resize the image with padded border
        new_unpad = int(round(shape[1] * ratio_)), int(round(shape[0] * ratio_))
        dw, dh = (
            new_shape[1] - new_unpad[0],
            new_shape[0] - new_unpad[1],
        )  # wh padding
        dw, dh = np.mod(dw, self.AUTO_SIZE), np.mod(
            dh, self.AUTO_SIZE
        )  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(
                self.img_orig, new_unpad, interpolation=cv2.INTER_LINEAR
            )
        else:
            img = self.img_orig.copy()
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=self.BORDER_COLOR,
        )  # add border
        return img

    def _prepare_tensor(self, img) -> torch.Tensor:
        # Convert the image to the expected format
        img_t = img_t[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_t = np.ascontiguousarray(img_t)
        img_t = torch.from_numpy(img_t).cpu()
        img_t = img_t.float() / 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)
        return img_t

    def prepare(self) -> torch.Tensor:
        self.names = self._load_classes()
        img = self._load_image()
        img_t = self._prepare_tensor(img)
        self.tensor_shape = img_t.shape
        return img_t

    def serve(self, inference_data: torch.Tensor) -> None:
        pred = non_max_suppression(
            inference_data[0],
            conf_thres=0.4,
            iou_thres=0.5,
            classes=None,
            agnostic=True,
        )

        if not hasattr(self, "img_orig"):
            raise RuntimeError("Input must be prepared first!")
        img_copy = self.img_orig.copy()

        results = []
        for det in pred:  # detections per image
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    self.tensor_shape[2:], det[:, :4], self.shape
                ).round()

                for *xyxy, conf, class_id in det:
                    det_x = float((xyxy[0] + xyxy[2]) / 2)  # x center
                    det_y = float((xyxy[1] + xyxy[3]) / 2)  # y center
                    det_w = float(xyxy[2] - xyxy[0])  # width
                    det_h = float(xyxy[3] - xyxy[1])  # height

                    results.append(
                        {
                            "class": self.names[int(class_id)],
                            "conf": float(conf),
                            "x": det_x,
                            "y": det_y,
                            "w": det_w,
                            "h": det_h,
                            "area": det_w
                            * det_h
                            / (self.shape[0] * self.shape[1]),
                        }
                    )

                    color = [random.randint(0, 255) for _ in range(3)]
                    cv2.rectangle(
                        img_copy,
                        (
                            int(det_x - det_w / 2.0),
                            int(det_y - det_h / 2.0),
                        ),
                        (
                            int(det_x + det_w / 2.0),
                            int(det_y + det_h / 2.0),
                        ),
                        color,
                        thickness=2,
                    )

                    text_size = cv2.getTextSize(
                        self.names[int(class_id)], 0, fontScale=0.5, thickness=1
                    )[0]
                    cv2.rectangle(
                        img_copy,
                        (
                            int(det_x - det_w / 2.0),
                            int(det_y - det_h / 2.0),
                        ),
                        (
                            int(det_x - det_w / 2.0 + text_size[0]),
                            int(det_y - det_h / 2.0 - text_size[1] - 3),
                        ),
                        color,
                        -1,
                    )
                    cv2.putText(
                        img_copy,
                        self.names[int(class_id)],
                        (
                            int(det_x - det_w / 2.0),
                            int(det_y - det_h / 2.0 - 2),
                        ),
                        0,
                        fontScale=0.5,
                        color=(255, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        results.sort(key=lambda x: x.get("area") or 0.0, reverse=True)
        json_file = self.prepared_file.with_suffix(".json")
        with json_file.open("w") as f:
            json.dump({"results": results}, f)

        if os.environ.get("API_DEBUG", False):
            cv2.imwrite(self.prepared_file.as_posix(), img_copy)


class AiDaemon(Daemon):
    def ai(self, input: AiInput) -> None:
        pid = os.fork()
        if pid != 0:
            return

        try:
            MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/app/yolov4.weights")

            # Load model
            model = Darknet(None, AiInput.IMAGE_SIZE).cpu()
            try:
                model.load_state_dict(
                    torch.load(MODEL_PATH, map_location="cpu")["model"]
                )
            except:
                model.load_darknet_weights(MODEL_PATH)
            model.eval()

            # Inference
            pred = model(input.prepare(), augment=False)
            input.serve(pred)

            input.update_metadata({"processed": "true"})
        except Exception as e:
            if os.environ.get("DEBUG", "false").lower() in ("true", "1", "on"):
                print(traceback.format_exc())
            input.update_metadata({"processed": "error"})

        sys.exit()


if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AiDaemon(input_type=AiInput, pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
