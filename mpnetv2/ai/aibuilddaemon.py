#!/usr/bin/env python3
"""
The document processor converts a document into a list of text items which in turn are passed through a machine learning
based on the MPNet v2 model. The model output is stored in Redis.
"""
import os
import traceback
from typing import Iterable, Optional

import torch
from redis import Redis
from redis.commands.search.field import NumericField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from transformers import AutoModel, AutoTokenizer

from daemon import AiBatchDaemon as Daemon

from .aibuildbatch import TextItem


class AiBuildDaemon(Daemon):
    @staticmethod
    def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean Pooling - Take attention mask into account for correct averaging
        :param model_output: Model output
        :param attention_mask: Attention mask
        :return: Mean pooled vector
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def create_redis_vector_index(self, search_space: Optional[str] = None) -> None:
        """
        Creates a RedisSearch index for the vector field of the text items. The index is named after the search space.
        :param search_space: Documents can be grouped into different search spaces, e.g. "en" or "de"
        """
        pieces = [os.getenv("DOC_PREFIX", "doc"), search_space or ""]
        index_name = ":".join([piece for piece in pieces if piece])
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
        """
        Loads the model and the tokenizer. Both are loaded from the MODEL_PATH environment variable.
        This method also creates a Redis client using the REDIS_HOST, REDIS_PORT and REDIS_DB environment variables.
        :envvar MODEL_PATH: Path to the model and tokenizer, e.g. "/opt/app/mpnet-base-v2"
        :envvar REDIS_HOST: Hostname of the Redis server. Default: "localhost"
        :envvar REDIS_PORT: Port of the Redis server. Default: 6379
        :envvar REDIS_DB: Redis database. Default: 0
        """
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
        """
        Checks the input file queue for new files. If a new file is found, it is read and split into text items.
        The text items are then passed through the machine learning model and the output is stored in Redis.
        """
        while input_batch := self.input_type.get_input_batch():
            model_input = self.input_type(input_batch, redis=self.redis)
            search_spaces = set(
                model_input.get_metadata(staged_file).get("search_space", "") for staged_file in input_batch
            )
            for search_space in search_spaces:
                self.create_redis_vector_index(search_space=search_space)
            for prepared_input in model_input.prepare(self.batch_size):
                model_output = self.ai(prepared_input)
                model_input.serve(model_output)

    def ai(self, input: Iterable[TextItem]) -> Optional[torch.Tensor]:
        """
        Method that passes the text items through the machine learning model and returns the model output as a tensor.
        :param input: Iterable of text items
        :return: Model output as tensor
        :envvar DEBUG: If set to "true", the stack trace of an exception is printed to stdout. Default: "false"
        """
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
