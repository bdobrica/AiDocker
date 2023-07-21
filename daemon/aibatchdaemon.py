import os
import signal
import time
from pathlib import Path
from typing import Type

from .aibatch import AiBatch
from .daemon import Daemon, PathLike

__version__ = "0.8.12"


class AiBatchDaemon(Daemon):
    def __init__(
        self,
        batch_type: Type[AiBatch],
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

        :param batch_type: The type of batch to use. This must be a subclass of AiBatch.
        :param pidfile: The path to where the file containing the pid should be stored.
        :param chroot: The path to the directory to chroot to.
        :param stdin: The path to the file to use as stdin.
        :param stdout: The path to the file to use as stdout.
        :param stderr: The path to the file to use as stderr.
        """
        self.batch_type = batch_type
        super().__init__(pidfile=pidfile, chroot=chroot, stdin=stdin, stdout=stdout, stderr=stderr)

    def ai(self, batch: AiBatch) -> None:
        """
        Method that is called when a batch is ready to be processed.

        :param batch: The batch to process.
        """
        raise NotImplementedError("You must implement ai(batch: AiBatch)")

    def queue(self) -> None:
        """
        Method that is called to check the staged folder for new files and construct a batch from them. The batch is
        then passed to the ai method. JSON files are ignored as they are used to pass metadata to the daemon.

        Environment variables:
        - STAGED_PATH: The path to the staged folder. Defaults to /tmp/ai/staged.
        - BATCH_SIZE: The number of files to process in a single batch. Defaults to 8.
        """
        STAGED_PATH = os.environ.get("STAGED_PATH", "/tmp/ai/staged")
        BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))

        staged_files = sorted(
            [f for f in Path(STAGED_PATH).glob("*") if f.is_file() and f.suffix != ".json"],
            key=lambda f: f.stat().st_mtime,
        )

        while staged_files:
            batch = self.batch_type(staged_files=staged_files[:BATCH_SIZE])
            staged_files = staged_files[BATCH_SIZE:]
            self.ai(batch)

    def run(self) -> None:
        """
        Method that is called when the daemon is started. This method is responsible for calling the queue method in a
        loop.

        Environment variables:
        - QUEUE_LATENCY: The number of seconds to wait between each call to the queue method. Defaults to 1.0.
        """
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        while True:
            self.queue()
            time.sleep(float(os.environ.get("QUEUE_LATENCY", 1.0)))
