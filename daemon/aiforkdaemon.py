import logging
import multiprocessing as mp
import os
import signal
import time
from typing import Any, Type

from .aiinput import AiInput
from .daemon import Daemon

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.getenv("DEBUG", "").lower() in ("1", "true", "yes") else logging.WARNING)


class AiForkDaemon(Daemon):
    def __init__(
        self,
        input_type: Type[AiInput],
        pidfile: os.PathLike,
        chroot: os.PathLike,
        stdin: os.PathLike = os.devnull,
        stdout: os.PathLike = os.devnull,
        stderr: os.PathLike = os.devnull,
    ):
        self.input_type = input_type
        super().__init__(pidfile=pidfile, chroot=chroot, stdin=stdin, stdout=stdout, stderr=stderr)

    @property
    def workers_number(self) -> int:
        return int(os.getenv("MAX_FORK", 8))

    def ai(self, input: Any) -> Any:
        raise NotImplementedError("You must implement ai(input: Any) -> Optional[Any]")

    def requeue_worker(self, worker_id: int) -> None:
        logger.info("Restarting worker %s", worker_id)
        self.workers_pool.append(worker_id)

    def fork_worker(self, worker_id: int, staged_batch: Any) -> None:
        logger.info("Starting worker %s processing %s", worker_id, staged_batch)

        model_input = self.input_type(staged_batch)
        model_output = self.ai(model_input.prepare())
        model_input.serve(model_output)

    def queue(self):
        mp_context = mp.get_context("fork")

        self.workers_pool = list(range(self.workers_number))

        with mp_context.Pool(processes=self.workers_number) as pool:
            while self.workers_pool and (input_batch := self.input_type.get_input_batch(batch_size=1)):
                worker_id = self.workers_pool.pop(0)
                pool.apply_async(self.fork_worker, [worker_id, input_batch], callback=self.requeue_worker)

    def run(self):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        while True:
            self.queue()
            time.sleep(float(os.getenv("QUEUE_LATENCY", 1.0)))
