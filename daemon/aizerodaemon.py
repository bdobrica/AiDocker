import logging

try:
    import torch.multiprocessing as mp
except ImportError:
    import multiprocessing as mp

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import zmq

from .aiforkdaemon import AiForkDaemon
from .aiinput import AiInput
from .zeroqueuemixin import ZeroQueueMixin

logger = logging.getLogger(__name__)


class AiZeroDaemon(AiForkDaemon, ZeroQueueMixin):
    def ai(self, input: AiInput) -> Dict[str, Any]:
        raise NotImplementedError("You must implement ai(input: AiInput) -> Dict[str, Any]")

    def worker_load(self) -> None:
        """
        Method called when a worker is initialized. For example, you can load a model here.
        """
        logger.info("Initializing worker %s", self.worker_id)

    def requeue_worker(self, worker_id: int) -> None:
        logger.info("Restarting worker %s", worker_id)
        self.workers_pool.append(worker_id)

    def zero_worker(self, worker_id: int) -> int:
        logger.info("Starting worker %s", worker_id)

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.connect(self.worker_address)

        self.worker_load()

        served_request: int = 1
        errors: int = 0

        while True:
            try:
                socket_payload = socket.recv_json()
                logging.debug("Received payload: %s", socket_payload)
                model_input = self.input_type(socket_payload)
                model_output = self.ai(model_input)
            except Exception as err:
                errors += 1
                logger.error("Error in worker %s: %s", worker_id, err)
                if errors > self.worker_errors:
                    logger.error("Too many errors in worker %s, exiting", worker_id)
                    break

            logger.debug("Sending payload: %s", model_output)
            socket.send_json({"worker_id": worker_id, **model_output})
            if self.worker_requests:
                served_request += 1
                if served_request > self.worker_requests:
                    logger.info("Worker %s served %s requests, exiting", worker_id, served_request)
                    break
            time.sleep(self.worker_latency)
        return worker_id

    def queue(self) -> None:
        mp_context = mp.get_context("fork")

        self.workers_pool = list(range(self.workers_number))

        with mp_context.Pool(processes=self.workers_number) as pool:
            while True:
                if self.workers_pool:
                    worker_id = self.workers_pool.pop(0)
                    pool.apply_async(self.zero_worker, [worker_id], callback=self.requeue_worker)
                time.sleep(self.worker_latency)
