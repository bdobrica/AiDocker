import zmq
from flask import Response, request

from .. import __version__
from .helpers import get_socket


def get_json() -> Response:
    socket = get_socket()
    socket.send_json(request.json)

    try:
        response = socket.recv_json()
    except zmq.error.Again:
        response = {"error": "timeout"}
        socket.close()

    response["version"] = __version__

    return Response(response, mimetype="application/json")
