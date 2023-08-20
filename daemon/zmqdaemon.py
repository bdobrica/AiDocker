import os
from pathlib import Path
from typing import Optional

import zmq

from .daemon import Daemon
from .zeroqueuemixin import ZeroQueueMixin


class ZMQDaemon(Daemon, ZeroQueueMixin):
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
