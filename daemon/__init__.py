import atexit
import json
import os
import signal
import sys
import time
from pathlib import Path

__version__ = "0.9.0"


class Daemon:
    def __init__(
        self,
        pidfile,
        chroot,
        stdin=os.devnull,
        stdout=os.devnull,
        stderr=os.devnull,
    ):
        self.pidfile = pidfile
        self.chroot = chroot
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr

    def daemonize(self):
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as error:
            sys.stderr.write(
                "First fork failed: {errno} ({error})\n".format(
                    errno=error.errno, error=error.strerror
                )
            )
            sys.exit(1)

        os.chdir(str(self.chroot))
        os.setsid()
        os.umask(0)

        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as error:
            sys.stderr.write(
                "Second fork failed: {errno} ({error})\n".format(
                    errno=error.errno, error=error.strerror
                )
            )
            sys.exit(1)

        sys.stdout.flush()
        sys.stderr.flush()
        si = open(self.stdin, "r")
        so = open(self.stdout, "a+")
        se = open(self.stderr, "a+")

        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())

        atexit.register(self.atexit)
        pid = str(os.getpid())

        pidfile_dir = Path(self.pidfile).parent
        if not pidfile_dir.is_dir():
            try:
                os.mkdir(str(pidfile_dir.absolute()))
            except:
                sys.stderr.write(
                    "Cannot create the pidfile directory {pidfile_dir}.\n".format(
                        pidfile_dir=str(pidfile_dir.absolute())
                    )
                )
                sys.exit(1)

        with open(self.pidfile, "w+") as fp:
            fp.write(pid + "\n")

    def atexit(self):
        self.delpid()

    def delpid(self):
        if Path(self.pidfile).is_file():
            os.remove(self.pidfile)

    def start(self):
        pid = None
        try:
            with open(self.pidfile, "r") as fp:
                pid = int(fp.read().strip())
        except IOError:
            pass

        if pid:
            sys.stderr.write(
                "Pidfile {pidfile} already exists. Daemon already running?\n".format(
                    pidfile=self.pidfile
                )
            )
            sys.exit(1)

        self.daemonize()
        self.run()

    def stop(self):
        pid = None
        try:
            with open(self.pidfile, "r") as fp:
                pid = int(fp.read().strip())
        except IOError:
            pass

        if not pid:
            sys.stderr.write(
                "Pidfile {pidfile} does not exist. Daemon not running?\n".format(
                    pidfile=self.pidfile
                )
            )
            sys.exit(1)

        try:
            while True:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.1)
        except OSError as error:
            error = str(error)
            if error.find("No such process") > 0:
                if Path(self.pidfile).is_file():
                    os.remove(self.pidfile)
            else:
                sys.stderr.write(
                    "Trying to kill the daemon resulted in error: {error}.\n".format(
                        error=error
                    )
                )
                sys.exit(1)

    def restart(self):
        self.stop()
        self.start()

    def run(self):
        while True:
            pass
        # raise NotImplementedError('You should overwrite this method.')


class ImageDaemon(Daemon):
    def queue(self):
        STAGED_PATH = os.environ.get("STAGED_PATH", "/tmp/ai/staged")
        SOURCE_PATH = os.environ.get("SOURCE_PATH", "/tmp/ai/source")
        PREPARED_PATH = os.environ.get("PREPARED_PATH", "/tmp/ai/prepared")
        MAX_FORK = int(os.environ.get("MAX_FORK", 8))
        CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 4096))

        staged_files = sorted(
            [
                f
                for f in Path(STAGED_PATH).glob("*")
                if f.is_file() and f.suffix != ".json"
            ],
            key=lambda f: f.stat().st_mtime,
        )
        source_files = [f for f in Path(SOURCE_PATH).glob("*") if f.is_file()]
        source_files_count = len(source_files)

        while source_files_count < MAX_FORK and staged_files:
            source_files_count += 1
            staged_file = staged_files.pop(0)

            meta_file = staged_file.with_suffix(".json")
            if meta_file.is_file():
                with meta_file.open("r") as fp:
                    try:
                        image_metadata = json.load(fp)
                    except:
                        image_metadata = {}
            image_metadata = {
                **{
                    "extension": staged_file.suffix,
                },
                **image_metadata,
            }

            source_file = Path(SOURCE_PATH) / staged_file.name
            prepared_file = Path(PREPARED_PATH) / (
                staged_file.stem + image_metadata["extension"]
            )

            with staged_file.open("rb") as src_fp, source_file.open(
                "wb"
            ) as dst_fp:
                while True:
                    chunk = src_fp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    dst_fp.write(chunk)

            staged_file.unlink()
            self.ai(source_file, prepared_file, **image_metadata)

    def run(self):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        while True:
            self.queue()
            time.sleep(1.0)
