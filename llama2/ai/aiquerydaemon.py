#!/usr/bin/env python3
import os
import traceback
from typing import Any, Dict

import numpy as np
import torch
from redis import Redis
from transformers import AutoModel, AutoTokenizer

from daemon import AiZeroDaemon as Daemon

from .aiqueryinput import AiQueryInput
from .llama import Llama


class AiQueryDaemon(Daemon):
    @staticmethod
    def mean_pooling(model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def worker_load(self) -> None:
        # Initialize
        MODEL_PATH = os.getenv("MODEL_PATH", "/opt/app/llama2")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. The LLAMA2 model requires CUDA.")

        # Load model
        self.model = Llama.build(
            ckpt_dir=os.path.join(MODEL_PATH, "llama-2-7b-chat"),
            tokenizer_path=os.path.join(MODEL_PATH, "tokenizer.model"),
            max_seq_len=int(os.getenv("LLAMA2_MAX_SEQ_LEN", 128)),
            max_batch_size=int(os.getenv("LLAMA2_MAX_BATCH_SIZE", 4)),
        )

    def ai(self, input: AiQueryInput) -> Dict[str, Any]:
        try:
            model_input = input.prepare()
            model_output = self.model.text_completion(
                model_input.messages,
                max_gen_len=model_input.max_gen_len,
                temperature=model_input.temperature,
                top_p=model_input.top_p,
            )

            return {
                "choices": [
                    {
                        "message": {
                            "content": model_output["generation"]["content"],
                            "role": model_output["generation"]["role"],
                        }
                    }
                ]
            }

        except Exception as e:
            if os.getenv("DEBUG", "false").lower() in ("true", "1", "on"):
                print(traceback.format_exc())
