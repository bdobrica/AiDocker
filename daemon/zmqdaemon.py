import os

import zmq

from .daemon import Daemon


class ZmqDaemon(Daemon):
    def run(self):
        context = zmq.Context()

        client_socket: zmq.Socket = context.socket(zmq.ROUTER)
        client_socket.bind(client_address)

        worker_socket: zmq.Socket = context.socket(zmq.DEALER)
        worker_socket.bind(worker_address)

        try:
            zmq.proxy(client_socket, worker_socket)
        finally:
            client_socket.close()
            worker_socket.close()
            context.term()
