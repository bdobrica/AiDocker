from flask import Flask

from ....daemon import ZeroQueueMixin
from .json import get_json
from .text import put_text


def load(app: Flask) -> None:
    """
    Initialize the ZeroMQ worker and client for the Flask app. This function
    is called by __main__.py to load the ZeroMQ worker and client for the Flask
    app iff the app requires a ZeroMQ worker.

    @param app: The Flask app to initialize ZeroMQ for.
    """

    with app.app_context():
        if "zmq_worker_timeout" in app.config and "zmq_client_address" in app.config:
            app.logger.info("ZeroMQ already loaded")
            return

        app.logger.info("Loading ZeroMQ")
        zmq_mixin = ZeroQueueMixin()
        app.config["zmq_worker_timeout"] = zmq_mixin.worker_timeout
        app.config["zmq_client_address"] = zmq_mixin.client_address


__all__ = [
    "get_json",
    "put_text",
    "load",
]
