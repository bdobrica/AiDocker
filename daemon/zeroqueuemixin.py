import os
from pathlib import Path
from typing import Optional


class ZeroQueueMixin:
    @property
    def workers_number(self) -> int:
        return int(os.getenv("MAX_FORK", 8))

    @property
    def worker_requests(self) -> int:
        return int(os.getenv("ZMQ_WORKER_REQUESTS", 0))

    @property
    def worker_latency(self) -> float:
        return float(os.getenv("ZMQ_WORKER_LATENCY", 0.01))

    @property
    def worker_errors(self) -> int:
        return int(os.getenv("ZMQ_WORKER_ERRORS", 0))

    @property
    def worker_timeout(self) -> int:
        return int(1000 * float(os.getenv("ZMQ_WORKER_TIMEOUT", "1.0")))

    @property
    def client_address(self) -> str:
        return self.get_zmq_address("ZMQ_CLIENT_SOCKET_PATH", "/tmp/ai/client")

    @property
    def worker_address(self) -> str:
        return self.get_zmq_address("ZMQ_WORKER_SOCKET_PATH", "/tmp/ai/worker")

    def get_zmq_address(self, env_var: str, default_value: Optional[str] = None) -> str:
        socket_path = Path(os.getenv(env_var, default_value)).with_suffix(self.model_suffix)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        return f"ipc://{socket_path.absolute().as_posix()}"
