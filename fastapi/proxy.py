#!/usr/bin/env python

import asyncio
import sys

import zmq
from zmq.asyncio import Context, Poller, ZMQEventLoop

from .config import Config


class Proxy:
    def __init__(self, client_socket: str, worker_socket: str):
        self.client_socket = client_socket
        self.worker_socket = worker_socket

        self.loop = ZMQEventLoop()
        asyncio.set_event_loop(self.loop)
        self.context = Context()

        self.client_ctx = self.context.socket(zmq.ROUTER)
        self.client_ctx.bind(self.client_socket)
        self.worker_ctx = self.context.socket(zmq.DEALER)
        self.worker_ctx.bind(self.worker_socket)

    @asyncio.coroutine
    def run_proxy(self):
        poller = Poller()
        poller.register(self.client_ctx, zmq.POLLIN)
        poller.register(self.worker_ctx, zmq.POLLIN)
        while True:
            events = yield from poller.poll()
            events = dict(events)
            if self.client_ctx in events:
                msg = yield from self.client_ctx.recv_multipart()
                yield from self.worker_ctx.send_multipart(msg)
            elif self.worker_ctx in events:
                msg = yield from self.worker_ctx.recv_multipart()
                yield from self.client_ctx.send_multipart(msg)

    def run(self):
        task = asyncio.ensure_future(self.run_proxy())
        self.loop.run_until_complete(asyncio.wait(task))


def main():
    proxy = Proxy(Config.CLIENT_SOCKET, Config.WORKER_SOCKET)
    proxy.run()


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    main()
