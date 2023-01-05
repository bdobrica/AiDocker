import atexit
import os
import sys
import time
from pathlib import Path
from signal import SIGTERM

__version__ = "0.8.9"


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

        self.load()

    def load(self):
        pass

    def daemonize(self):
        if os.environ.get("DEBUG", "false").lower() in ("true", "1", "on"):
            return

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
                os.kill(pid, SIGTERM)
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
