import json
import time

import zmq
from flask import current_app, request

from .. import __version__
from ..wrappers import ApiResponse
from .helpers import Socket


def get_json() -> ApiResponse:
    with Socket() as socket:
        current_app.logger.debug("Sending request: %s", request.json)
        socket.send_json(request.json)

        start_time = time.perf_counter()
        try:
            response = socket.recv_json()
            current_app.logger.debug("Received response: %s", response)
        except zmq.error.Again:
            response = {"error": "timeout"}
            socket.close()
            current_app.logger.debug("Worker connection timeout")

        if not isinstance(response, dict):
            response = {"error": "invalid response"}

        response["version"] = __version__
        response["latency"] = "%.2f" % (time.perf_counter() - start_time)

        return ApiResponse.from_dict(response)
