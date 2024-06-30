import os

from .aiquerydaemon import AiQueryDaemon
from .aiqueryinput import AiQueryInput

if __name__ == "__main__":
    CHROOT_PATH = os.getenv("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.getenv("PIDFILE_PATH", f"/opt/app/run/fasttext.pid")

    AiQueryDaemon(input_type=AiQueryInput, pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
