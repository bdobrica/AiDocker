#!/usr/bin/env python3
import itertools
import json
import os
import traceback
from pathlib import Path
from typing import Generator, List

import numpy as np
import torch
from redis import Redis
from transformers import AutoModel, AutoTokenizer

from daemon import AiBatch as Batch
from daemon import AiBatchDaemon as Daemon

from .reader import TextItem, read_text

__version__ = "0.8.13"


class AiBatch(Batch):
    def get_text_item(self) -> Generator[TextItem]:
        for source_file in self.source_files:
            yield from read_text(Path(source_file))

    def prepare(self) -> Generator[List[TextItem]]:
        yield list(itertools.islice(self.get_text_item(), 0, int(os.environ.get("BATCH_SIZE", 8))))

    def serve(self, inference_data: np.ndarray) -> None:
        with open(self.target_file, "w") as fp:
            json.dump(inference_data.tolist(), fp)


class AiDaemon(Daemon):
    def mean_pooling(model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def load(self) -> None:
        # Initialize
        MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/app/mpnet-base-v2")
        MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cpu")
        if MODEL_DEVICE == "cuda" and not torch.cuda.is_available():
            MODEL_DEVICE = "cpu"
        self.device = torch.device(MODEL_DEVICE)

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModel.from_pretrained(MODEL_PATH)
        self.redis = Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            db=int(os.environ.get("REDIS_DB", 0)),
        )

    def ai(self, batch: AiBatch) -> None:
        try:
            for model_input in batch.prepare():
                encoded_input_t = self.tokenizer(
                    [item.text for item in model_input], padding=True, truncation=True, return_tensors="pt"
                )
                with torch.no_grad():
                    model_output_t = self.model(**encoded_input_t)
                model_output_t = AiDaemon.mean_pooling(model_output_t, encoded_input_t["attention_mask"])
                model_output = model_output_t.cpu().numpy()
                for item, vector in zip(model_input, model_output):
                    item.store(self.redis, vector.flatten())
            batch.update_metadata({"processed": "true"})

        except Exception as e:
            batch.update_metadata({"processed": "error"})
            if os.environ.get("DEBUG", "false").lower() in ("true", "1", "on"):
                print(traceback.format_exc())


if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AiDaemon(batch_type=AiBatch, pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
