"""
This module provides a generic daemon class. Subclass the Daemon class and override the run() method.

Inspired by: https://lloydrochester.com/post/c/unix-daemon-example/#c-code-double-fork-daemon-example

Example usage:

    ```python
    from daemon import Daemon

    class MyDaemon(Daemon):
        def run(self) -> None:
            while True:
                do_something()
    
    if __name__ == "__main__":
        daemon = MyDaemon(pidfile="/tmp/mydaemon.pid", chroot="/")
        if len(sys.argv) == 2:
            if "start" == sys.argv[1]:
                daemon.start()
            elif "stop" == sys.argv[1]:
                daemon.stop()
            elif "restart" == sys.argv[1]:
                daemon.restart()
            else:
                print("unknown option %s" % sys.argv[1])
                sys.exit(1)
            sys.exit(0)
        else:
            print("usage: %s start|stop|restart" % sys.argv[0])
    ```
"""
import atexit
import os
import sys
import time
from pathlib import Path
from signal import SIGTERM
from typing import Union

__version__ = "0.8.12"

PathLike = Union[str, Path]


class Daemon:
    def __init__(
        self,
        pidfile: PathLike,
        chroot: PathLike,
        stdin: PathLike = os.devnull,
        stdout: PathLike = os.devnull,
        stderr: PathLike = os.devnull,
    ) -> None:
        """
        Initialize the daemon.
        :param pidfile: The path to where the file containing the pid should be stored.
        :param chroot: The path to the directory to chroot to.
        :param stdin: The path to the file to use as stdin.
        :param stdout: The path to the file to use as stdout.
        :param stderr: The path to the file to use as stderr.
        """
        self.pidfile = pidfile
        self.chroot = chroot
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr

        self.load()

    def load(self) -> None:
        """
        This method is called after the daemon is initialized, but before it is daemonized.
        Allows for loading of any resources that should be available to the daemon and its children. Intended to be
        overwritten by subclasses. Use env-vars to pass data to the daemon.
        """
        pass

    def daemonize(self) -> None:
        """
        Daemonize the current process.
        Setting the DEBUG environment variable to true will prevent the process from being daemonized,
        allowing for easier debugging.
        """
        if os.environ.get("DEBUG", "false").lower() in ("true", "1", "on"):
            return

        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as error:
            sys.stderr.write("First fork failed: {errno} ({error})\n".format(errno=error.errno, error=error.strerror))
            sys.exit(1)

        os.chdir(str(self.chroot))
        os.setsid()
        os.umask(0)

        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as error:
            sys.stderr.write("Second fork failed: {errno} ({error})\n".format(errno=error.errno, error=error.strerror))
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
                pidfile_dir.mkdir(parents=True, exist_ok=True)
            except:
                sys.stderr.write(
                    "Cannot create the pidfile directory {pidfile_dir}.\n".format(
                        pidfile_dir=str(pidfile_dir.absolute())
                    )
                )
                sys.exit(1)

        with open(self.pidfile, "w+") as fp:
            fp.write(pid + "\n")

    def atexit(self) -> None:
        """
        Method to be called when the daemon exits.
        In this case, it is used to remove the pidfile.
        """
        self.delpid()

    def delpid(self) -> None:
        """
        Remove the pidfile if it exists.
        """
        if Path(self.pidfile).is_file():
            os.remove(self.pidfile)

    def start(self) -> None:
        """
        Method to start the daemon.
        This method will check if the pidfile exists, and if it does, it will print an error and exit to prevent
        multiple instances of the daemon from running.

        The method will then call the daemonize() method that will double-fork the daemon, followed by calling the
        run() method.
        """
        pid = None
        try:
            with open(self.pidfile, "r") as fp:
                pid = int(fp.read().strip())
        except IOError:
            pass

        if pid:
            sys.stderr.write("Pidfile {pidfile} already exists. Daemon already running?\n".format(pidfile=self.pidfile))
            sys.exit(1)

        self.daemonize()
        self.run()

    def stop(self) -> None:
        """
        Method to stop the daemon.
        This method will check if the pidfile exists, and if it does, it will attempt to kill the process with the pid
        in the pidfile. If the process does not exist, it will remove the pidfile.
        """
        pid = None
        try:
            with open(self.pidfile, "r") as fp:
                pid = int(fp.read().strip())
        except IOError:
            pass

        if not pid:
            sys.stderr.write("Pidfile {pidfile} does not exist. Daemon not running?\n".format(pidfile=self.pidfile))
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
                sys.stderr.write("Trying to kill the daemon resulted in error: {error}.\n".format(error=error))
                sys.exit(1)

    def restart(self) -> None:
        """
        Method to restart the daemon.
        Calls the stop() method, followed by the start() method.
        """
        self.stop()
        self.start()

    def run(self):
        """
        Method to be overwritten by subclasses.
        This method will be called after the daemon is daemonized.
        """
        while True:
            pass
        # raise NotImplementedError('You should overwrite this method.')
