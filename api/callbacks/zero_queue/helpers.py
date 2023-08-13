import zmq
from flask import current_app


def get_socket() -> zmq.Socket:
    current_app.logger.debug("Creating ZeroMQ socket")
    context = zmq.Context()
    worker_timeout = current_app.config["zmq_worker_timeout"]
    client_address = current_app.config["zmq_client_address"]
    current_app.logger.debug("Connecting to ZeroMQ socket at %s", client_address)
    socket: zmq.Socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, worker_timeout)
    socket.connect(client_address)
    return socket
