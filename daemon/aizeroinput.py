import json
import time
from typing import Any, Dict


class AiZeroInput:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload

    def prepare(self) -> Any:
        raise NotImplementedError("You must implement prepare() -> Any")

    def serve(self, inference_data: Any) -> Dict[str, Any]:
        raise NotImplementedError("You must implement serve(inference_data: any) -> Dict[str, Any]")
