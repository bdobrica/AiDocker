import atexit
import sys
from functools import cached_property

import zmq
from flask import current_app


class Socket:
    @cached_property
    def _context(self) -> zmq.Context:
        main_module = sys.modules["__main__"]
        if not hasattr(main_module, "_context"):
            setattr(main_module, "_context", zmq.Context())

            def cleanup() -> None:
                main_module = sys.modules["__main__"]
                if hasattr(main_module, "_context"):
                    main_module._context.term()

            atexit.register(cleanup)
        return main_module._context

    def __enter__(self) -> zmq.Socket:
        worker_timeout = current_app.config["zmq_worker_timeout"]
        client_address = current_app.config["zmq_client_address"]

        self._socket: zmq.Socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, worker_timeout)
        self._socket.setsockopt(zmq.SNDTIMEO, worker_timeout)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(client_address)
        return self._socket

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._socket.close()
