import zmq
from flask import Response, current_app, request

from .. import __version__
from .helpers import get_socket


def put_text() -> Response:
    current_app.logger.debug("Received request: %s", request.json)
    socket = get_socket()
    current_app.logger.debug("Sending request: %s", request.json)
    socket.send_json(request.json)

    try:
        response = socket.recv_json()
        current_app.logger.debug("Received response: %s", response)
    except zmq.error.Again:
        response = {"error": "timeout"}
        socket.close()
        current_app.logger.debug("Worker connection timeout")

    response["version"] = __version__

    return Response(response, mimetype="application/json")