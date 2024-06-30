#!/usr/bin/env python3
import os
import traceback
from typing import Any, Dict

import numpy as np
import openai
import torch
from redis import Redis
from transformers import AutoModel, AutoTokenizer

from daemon import AiZeroDaemon as Daemon

from .aiqueryinput import AiQueryInput
from .reader import TextItem


class AiQueryDaemon(Daemon):
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

    def worker_load(self) -> None:
        """
        A method that is called once when the worker is started. It loads the model and the tokenizer.
        :envvar MODEL_PATH: Path to the model, e.g. "/opt/app/mpnet-base-v2"
        :envvar MODEL_DEVICE: Device to run the model on, e.g. "cpu" or "cuda"
        :envvar REDIS_HOST: Hostname of the Redis server, e.g. "localhost"
        :envvar REDIS_PORT: Port of the Redis server, e.g. 6379
        :envvar REDIS_DB: Redis database, e.g. 0
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

    def call_chat_bot(self, query: str, search_results: str) -> str:
        """
        Calls the OpenAI chat bot to generate an answer to the query. The search results are passed as additional
        information to the chat bot.
        :param query: Query to be answered (natural language), e.g. "What is the capital of Germany?"
        :param search_results: Search results that are passed to the chat bot (natural language)
        :return: Answer to the query (natural language), e.g. "Berlin"
        """
        if openai.api_key is None:
            openai.api_key = os.getenv("OPENAI_API_KEY")

        if openai.api_key is None:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")

        completion = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a helpful assistant. You are helping a student with their homework. The student asks you a question and you need to answer it. Compose a comprehensive reply to the query using the following search results:\n{search_results}\nCite each reference using [number] notation (every result has this number at the beginning). Citation should be done at the end of each sentence. If the search results mention multiple subjects with the same name, create separate answers for each. Only include information found in the results and don't add any additional information. Make sure the answer is correct and don't output false content. If the text does not relate to the query, simply state 'Found Nothing'.""",
                },
                {"role": "user", "content": query},
            ],
        )

        return completion.choices[0].message

    def ai(self, input: AiQueryInput) -> Dict[str, Any]:
        """
        The main method of the worker. It's applied to the AiQueryInput object and returns a dictionary with the
        results. Here's an example of the returned dictionary:
        ```json
        {
            "results": [
                {
                    "text": "What is the capital of Germany?",
                    "matches": [
                        {
                            "text": "Berlin",
                            "score": 0.9
                        },
                        {
                            "text": "Munich",
                            "score": 0.8
                        }
                    ],,
                    "answer": "Berlin"
                }
            ]
        }
        ```
        :param input: AiQueryInput object
        :return: Dictionary with the results
        """
        try:
            model_inputs = input.prepare()
            encoded_input_t = self.tokenizer(
                [item.text for item in model_inputs], padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                model_output_t = self.model(**encoded_input_t)
            model_output_t = self.mean_pooling(model_output_t, encoded_input_t["attention_mask"])
            model_output: np.ndarray = model_output_t.cpu().numpy()

            results = []
            for model_input, vector in zip(model_inputs, model_output):
                matches = model_input.match(self.redis, vector)
                results.append(
                    {
                        "text": model_input.text,
                        "matches": sorted(
                            [{"text": item.text, "score": item.score} for item in matches],
                            key=lambda item: item["score"],
                            reverse=True,
                        ),
                        "answer": self.call_chat_bot(
                            query=model_input.text,
                            search_results="\n".join(
                                [f"{(item.page or item.paragraph)} {item.text}" for item in matches]
                            ),
                        ),
                    }
                )

            return {"results": results}

        except Exception as e:
            if os.getenv("DEBUG", "false").lower() in ("true", "1", "on"):
                print(traceback.format_exc())
