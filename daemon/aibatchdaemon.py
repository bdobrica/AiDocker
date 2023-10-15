"""
This module contains the AiBatchDaemon class, which is a type of daemon that is designed to extract batches of files
from a file queue and process them.
"""
import logging
import os
import time
from typing import Any, Optional, Type

from .aibatch import AiBatch
from .daemon import Daemon, PathLike

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.getenv("DEBUG", "").lower() in ("1", "true", "yes") else logging.WARNING)


class AiBatchDaemon(Daemon):
    def __init__(
        self,
        input_type: Type[AiBatch],
        pidfile: PathLike,
        chroot: PathLike,
        stdin: PathLike = os.devnull,
        stdout: PathLike = os.devnull,
        stderr: PathLike = os.devnull,
    ) -> None:
        """
        Initialize a type of daemon that is designed to process batches of requests.
        AiDocker containers are using a type of file queue to pass requests to the daemon and get AI results back. The
        file queue consists of 3 folders: staged, source and prepared. The staged folder contains the files that are
        received from the API and are ready to be processed. The AiBatch is responsible for moving the files from
        staged to source and then source to prepared. The source folder contains the files that are currently being
        processed. The prepared folder contains the files that have been processed and are ready to be sent back to the
        API.

        This daemon has a queue method that is responsible for checking the staged folder for new files and
        constructing a batch from them. The batch is then passed to the ai method, which is responsible for processing
        the batch.

        :param input_type: The type of batch to use. This must be a subclass of AiBatch.
        :param pidfile: The path to where the file containing the pid should be stored.
        :param chroot: The path to the directory to chroot to.
        :param stdin: The path to the file to use as stdin.
        :param stdout: The path to the file to use as stdout.
        :param stderr: The path to the file to use as stderr.
        """
        self.input_type = input_type
        super().__init__(pidfile=pidfile, chroot=chroot, stdin=stdin, stdout=stdout, stderr=stderr)

    @property
    def batch_size(self) -> int:
        return int(os.getenv("BATCH_SIZE", 8))

    def ai(self, input: Any) -> Optional[Any]:
        """
        Method that is called when a batch is ready to be processed.

        :param batch: The batch to process.
        """
        raise NotImplementedError("You must implement ai(batch: Any) -> Optional[Any]")

    def queue(self) -> None:
        """
        Method that is called to check the staged folder for new files and construct a batch from them. The batch is
        then passed to the ai method. JSON files are ignored as they are used to pass metadata to the daemon.

        Environment variables:
        - STAGED_PATH: The path to the staged folder. Defaults to /tmp/ai/staged.
        - BATCH_SIZE: The number of files to process in a single batch. Defaults to 8.
        """
        while input_batch := self.input_type.get_input_batch(self.batch_size):
            model_input = self.input_type(input_batch)
            model_output = self.ai(model_input.prepare())
            model_input.serve(model_output)

    def run(self) -> None:
        """
        Method that is called when the daemon is started. This method is responsible for calling the queue method in a
        loop.

        Environment variables:
        - QUEUE_LATENCY: The number of seconds to wait between each call to the queue method. Defaults to 1.0.
        """
        while True:
            self.queue()
            time.sleep(float(os.getenv("QUEUE_LATENCY", 1.0)))
