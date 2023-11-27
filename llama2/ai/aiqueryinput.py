from typing import Any, Dict, List, Optional, get_args

from daemon import AiZeroInput

from .llama.generation import Role


class AiQueryInput(AiZeroInput):
    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__(payload)
        self.messages: List[Dict[Role, str]] = []
        self.max_gen_len: Optional[int] = None
        self.temperature: float = 0.6
        self.top_p: float = 0.9

    def prepare(self) -> "AiQueryInput":
        self.max_gen_len = int(self.payload["max_gen_len"]) if "max_gen_len" in self.payload else None
        self.temperature = float(self.payload.get("temperature", 0.8))
        self.top_p = float(self.payload.get("top_p", 0.9))

        self.messages = []
        roles = get_args(Role)
        for message in self.payload.get("messages", []):
            if "role" not in message or message["role"] not in roles or "content" not in message:
                continue
            self.messages.append({message["role"]: message["content"]})

        return self
