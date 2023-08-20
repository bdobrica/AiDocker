#!/usr/bin/env python3
import os
import traceback
from typing import Iterable, Optional

import torch
from redis import Redis
from redis.commands.search.field import NumericField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from transformers import AutoModel, AutoTokenizer

from daemon import AiBatchDaemon as Daemon
from daemon import FileQueueMixin

from .aibuildbatch import TextItem


class AiBuildDaemon(Daemon, FileQueueMixin):
    @staticmethod
    def mean_pooling(model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def create_redix_vector_index(self, search_space: Optional[str] = None) -> None:
        index_name = [
            item
            for item in [
                self.model_name,
                os.environ.get("DOC_PREFIX", "doc"),
                search_space or "",
            ]
            if item
        ].join(":")
        try:
            self.redis.ft(index_name).info()
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
            definition = IndexDefinition(prefix=[f"{index_name}:"], index_type=IndexType.HASH)
            self.redis.ft(index_name).create_index(fields=schema, definition=definition)

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

    def queue(self) -> None:
        while staged_files := self.input_type.get_input_files():
            search_spaces = set(self.get_metadata(staged_file).get("search_space", "") for staged_file in staged_files)
            for search_space in search_spaces:
                self.create_redix_vector_index(search_space=search_space)
            model_input = self.input_type(staged_files, redis=self.redis)
            for prepared_input in model_input.prepare(self.batch_size):
                model_output = self.ai(prepared_input)
                model_input.serve(model_output)

    def ai(self, input: Iterable[TextItem]) -> Optional[torch.Tensor]:
        try:
            encoded_input_t = self.tokenizer(
                [item.text for item in input], padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                model_output_t = self.model(**encoded_input_t)
            model_output_t = self.mean_pooling(model_output_t, encoded_input_t["attention_mask"])
            return model_output_t
        except Exception as e:
            if os.getenv("DEBUG", "false").lower() in ("true", "1", "on"):
                print(traceback.format_exc())
            return None
