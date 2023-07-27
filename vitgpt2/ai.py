#!/usr/bin/env python3
import json
import os
import random
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
)

from daemon import AiBatch as Batch
from daemon import AiBatchDaemon as Daemon

__version__ = "0.8.13"


class AiBatch(Batch):
    def prepare(self) -> np.ndarray:
        # Load the images
        images = [cv2.cvtColor(cv2.imread(str(source_file)), cv2.COLOR_BGR2RGB) for source_file in self.source_files]

        # Compute maximum image size
        self.shapes = [im.shape for im in images]
        self.im_h = max([shape[0] for shape in self.shapes])
        self.im_w = max([shape[1] for shape in self.shapes])

        # Augment images
        images = [cv2.resize(im, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR) for im in images]

        # Parameters
        self.max_length = max([int(metadata.get("max_length", 16)) for metadata in self.metadata])
        self.num_beams = max([int(metadata.get("num_beams", 4)) for metadata in self.metadata])

        # Build batch input
        return np.stack(images)

    def serve(self, inference_data: np.ndarray) -> None:
        # Save the results
        for file_idx, prediction in enumerate(inference_data):
            result = {"caption": prediction.strip()}
            with open(self.prepared_files[file_idx], "w") as fp:
                json.dump(result, fp)


class AiDaemon(Daemon):
    def load(self) -> None:
        MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cuda:0")
        if MODEL_DEVICE.startswith("cuda") and not torch.cuda.is_available():
            MODEL_DEVICE = "cpu"
        self.device = torch.device(MODEL_DEVICE)

        MODEL_PATH = os.getenv("MODEL_PATH", "/opt/app/vitgpt2")

        # The trick is to not initialize the model outside the forked process
        # It saves some time, but not much. At least no disk I/O
        model_config = VisionEncoderDecoderConfig.from_pretrained(MODEL_PATH, local_files_only=True)
        model_sd = torch.load(
            (Path(MODEL_PATH) / "pytorch_model.bin").as_posix(),
            map_location=self.device,
        )
        feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_PATH, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model = VisionEncoderDecoderModel(config=model_config)
        self.model.to(self.device)
        self.model.load_state_dict(model_sd)

    def ai(self, batch: AiBatch) -> None:
        try:
            # Try reproducing the results
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

            # Inference
            model_input = batch.prepare()
            gen_kwargs = {
                "max_length": batch.max_length,
                "num_beams": batch.num_beams,
            }
            pixel_values = self.feature_extractor(images=model_input, return_tensors="pt").pixel_values
            output_ids = self.model.generate(pixel_values, **gen_kwargs)
            predictions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            batch.serve(predictions)
            batch.update_metadata({"processed": "true"})

        except Exception as e:
            batch.update_metadata({"processed": "error"})
            if os.getenv("DEBUG", "false").lower() in ("true", "1", "on"):
                print(traceback.format_exc())


if __name__ == "__main__":
    CHROOT_PATH = os.getenv("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.getenv("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AiDaemon(batch_type=AiBatch, pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
