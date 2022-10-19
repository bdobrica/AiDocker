from pathlib import Path

import numpy as np
import onnxruntime

from .detector_utils import preprocess_image

CACHED_FILES = {
    "default": {
        "checkpoint": "detector_v2_default_checkpoint.onnx",
        "classes": "detector_v2_default_classes",
    },
    "base": {
        "checkpoint": "detector_v2_base_checkpoint.onnx",
        "classes": "detector_v2_base_classes",
    },
}


class Detector:
    detection_model = None
    classes = None

    def __init__(self, model_name="default", app_path="/opt/app"):
        """
        model = Detector()
        """
        if model_name not in CACHED_FILES:
            raise ValueError(f"Model {model_name} not found")

        checkpoint_path = (
            Path(app_path) / CACHED_FILES[model_name]["checkpoint"]
        )
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint {checkpoint_path} not found")

        classes_path = Path(app_path) / CACHED_FILES[model_name]["classes"]
        if not classes_path.is_file():
            raise ValueError(f"Classes {classes_path} not found")

        self.detection_model = onnxruntime.InferenceSession(
            checkpoint_path.as_posix()
        )

        with open(classes_path, "r") as fp:
            self.classes = [c.strip() for c in fp.readlines() if c.strip()]

    def detect(self, img_path, mode="default", min_prob=None):
        if mode == "fast":
            image, scale = preprocess_image(
                img_path, min_side=480, max_side=800
            )
            if not min_prob:
                min_prob = 0.5
        else:
            image, scale = preprocess_image(img_path)
            if not min_prob:
                min_prob = 0.6

        outputs = self.detection_model.run(
            [s_i.name for s_i in self.detection_model.get_outputs()],
            {
                self.detection_model.get_inputs()[0].name: np.expand_dims(
                    image, axis=0
                )
            },
        )

        labels = [op for op in outputs if op.dtype == "int32"][0]
        scores = [op for op in outputs if isinstance(op[0][0], np.float32)][0]
        boxes = [op for op in outputs if isinstance(op[0][0], np.ndarray)][0]

        boxes /= scale
        processed_boxes = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < min_prob:
                continue
            box = box.astype(int).tolist()
            label = self.classes[label]
            processed_boxes.append(
                {
                    "box": [int(c) for c in box],
                    "score": float(score),
                    "label": label,
                }
            )

        return processed_boxes
