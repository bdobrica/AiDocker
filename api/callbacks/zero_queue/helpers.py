import zmq
from flask import g


def get_socket() -> zmq.Socket:
    context: zmq.Context = g.zmq_context
    client_address: str = g.zmq_client_address
    worker_timeout: int = g.zmq_worker_timeout
    socket: zmq.Socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, worker_timeout)
    socket.connect(client_address)
    return socket
