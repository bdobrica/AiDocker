import os

import zmq
from flask import Response, _app_ctx_stack, request

from .. import __version__


def put_text() -> Response:
    context: zmq.Context = getattr(_app_ctx_stack.top, "zmq_context")
    client_address: str = getattr(_app_ctx_stack.top, "zmq_client_address")
    socket: zmq.Socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, int(1000 * float(os.getenv("ZMQ_TIMEOUT", "1.0"))))
    socket.connect(client_address)
    socket.send_json(request.json)

    try:
        response = socket.recv_json()
    except zmq.error.Again:
        response = {"error": "timeout"}
        socket.close()

    response["version"] = __version__

    return Response(response, mimetype="application/json")
