import argparse
import os

from .aibuildbatch import AiBuildBatch
from .aibuilddaemon import AiBuildDaemon
from .aiquerydaemon import AiQueryDaemon
from .aiqueryinput import AiQueryInput

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Daemon")
    parser.add_argument("--daemon", "-d", type=str, help="daemon type")
    args = parser.parse_args()

    CHROOT_PATH = os.getenv("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.getenv("PIDFILE_PATH", "/opt/app/run/ai.pid")

    if args.daemon.lower() in ("aibuilddaemon", "builddaemon", "build"):
        AiBuildDaemon(batch_type=AiBuildBatch, pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
    elif args.daemon.lower() in ("aiquerydaemon", "querydaemon", "query"):
        AiQueryDaemon(batch_type=AiQueryInput, pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
    else:
        raise ValueError(f"Unknown daemon type: {args.daemon}")
