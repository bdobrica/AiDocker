import zmq
from flask import current_app


def get_socket() -> zmq.Socket:
    context = zmq.Context()
    worker_timeout = current_app.config["zmq_worker_timeout"]
    client_address = current_app.config["zmq_client_address"]
    socket: zmq.Socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, worker_timeout)
    socket.connect(client_address)
    return socket
