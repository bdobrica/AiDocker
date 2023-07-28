import os
from pathlib import Path
from typing import Optional

import yaml
import zmq

from .daemon import Daemon


class ZMQDaemon(Daemon):
    @property
    def model_suffix(self) -> str:
        return "." + self.config.get("model", "model")

    @property
    def config(self) -> dict:
        config_path = Path("/opt/app/container.yaml")
        with config_path.open("r") as fp:
            return yaml.safe_load(fp)

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

    def run(self):
        context = zmq.Context()

        client_socket: zmq.Socket = context.socket(zmq.ROUTER)
        client_socket.bind(self.client_address)

        worker_socket: zmq.Socket = context.socket(zmq.DEALER)
        worker_socket.bind(self.worker_address)

        try:
            zmq.proxy(client_socket, worker_socket)
        finally:
            client_socket.close()
            worker_socket.close()
            context.term()
