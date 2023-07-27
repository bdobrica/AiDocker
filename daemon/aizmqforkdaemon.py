import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Optional

import yaml
import zmq

from .aiforkdaemon import AiForkDaemon

logger = logging.getLogger(__name__)


class AiZMQForkDaemon(AiForkDaemon):
    @property
    def workers_number(self) -> int:
        return int(os.getenv("MAX_FORK", 8))

    @property
    def worker_requests(self) -> int:
        return int(os.getenv("ZMQ_WORKER_REQUESTS", 0))

    @property
    def worker_latency(self) -> float:
        return float(os.getenv("ZMQ_WORKER_LATENCY", 0.01))

    @property
    def worker_errors(self) -> int:
        return int(os.getenv("ZMQ_WORKER_ERRORS", 0))

    @property
    def model_suffix(self) -> str:
        return "." + self.config.get("model", "model")

    @property
    def config(self) -> dict:
        config_path = Path("/opt/app/container.yaml")
        with config_path.open("r") as fp:
            return yaml.safe_load(fp)

    @property
    def client_address(self) -> str:
        return self.get_zmq_address("ZMQ_CLIENT_SOCKET_PATH", "/tmp/ai/client")

    @property
    def worker_address(self) -> str:
        return self.get_zmq_address("ZMQ_WORKER_SOCKET_PATH", "/tmp/ai/worker")

    def get_zmq_address(self, env_var: str, default_value: Optional[str] = None) -> str:
        socket_path = Path(os.getenv(env_var, default_value)).with_suffix(self.model_suffix)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        return f"ipc://{socket_path.absolute().as_posix()}"

    def restart_worker(self, worker_id: int) -> None:
        logger.info(f"Restarting worker %s", worker_id)
        self.workers_pool.append(worker_id)

    def start_worker(self, worker_id: int) -> None:
        logger.info(f"Starting worker %s", worker_id)

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.connect(self.worker_address)

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

    def zero_queue(self) -> None:
        mp_context = mp.get_context("fork")

        self.workers_pool = list(range(self.workers_number))

        with mp_context.Pool(processes=self.workers_number) as pool:
            while True:
                if self.workers_pool:
                    worker_id = self.workers_pool.pop(0)
                    pool.apply_async(self.start_worker, [worker_id], callback=self.restart_worker)
                time.sleep(self.worker_latency)
