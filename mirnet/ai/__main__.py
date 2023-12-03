import argparse
import os

from .aibatch import AiBatch
from .aidaemon import AiDaemon

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Daemon")
    args = parser.parse_args()

    CHROOT_PATH = os.getenv("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.getenv("PIDFILE_PATH", f"/opt/app/run/mirnet.pid")

    AiDaemon(input_type=AiBatch, pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
