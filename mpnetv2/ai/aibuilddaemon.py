#!/usr/bin/env python3
import os
import traceback

import numpy as np
import torch
from redis import Redis
from redis.commands.search.field import NumericField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from transformers import AutoModel, AutoTokenizer

from daemon import AiDaemon as Daemon

from .aibuildbatch import AiBuildBatch


class AiBuildDaemon(Daemon):
    def mean_pooling(model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def create_redix_vector_index(self) -> None:
        try:
            self.redis.ft(self.model_name).info()
        except:
            schema = (
                TextField("text", weight=1.0),
                NumericField("page"),
                NumericField("paragraph"),
                TextField("path"),
                VectorField(
                    "vector",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": 768,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )
            definition = IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH)
            self.redis.ft(self.model_name).create_index(fields=schema, definition=definition)

    def load(self) -> None:
        # Initialize
        MODEL_PATH = os.getenv("MODEL_PATH", "/opt/app/mpnet-base-v2")
        MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")
        if MODEL_DEVICE == "cuda" and not torch.cuda.is_available():
            MODEL_DEVICE = "cpu"
        self.device = torch.device(MODEL_DEVICE)

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModel.from_pretrained(MODEL_PATH)
        self.redis = Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
        )
        self.create_redix_vector_index()

    def ai(self, batch: AiBuildBatch) -> None:
        try:
            model_output = None
            for model_input in batch.prepare():
                encoded_input_t = self.tokenizer(
                    [item.text for item in model_input], padding=True, truncation=True, return_tensors="pt"
                )
                with torch.no_grad():
                    model_output_t = self.model(**encoded_input_t)
                model_output_t = self.mean_pooling(model_output_t, encoded_input_t["attention_mask"])
                if model_output is None:
                    model_output = model_output_t.cpu().numpy()
                else:
                    model_output = np.vstack((model_output, model_output_t.cpu().numpy()))
                model_output = model_output_t.cpu().numpy()
                for item, vector in zip(model_input, model_output):
                    item.store(self.redis, vector.flatten())
            batch.update_metadata({"processed": "true"})

        except Exception as e:
            batch.update_metadata({"processed": "error"})
            if os.getenv("DEBUG", "false").lower() in ("true", "1", "on"):
                print(traceback.format_exc())
