import argparse
import os
from pathlib import Path

from .queuecleaner import QueueCleaner
from .zmqdaemon import ZMQDaemon

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", "-d", type=str, default="queuecleaner", help="Daemon to control")
    parser.add_argument("--action", "-a", type=str, default="start", help="Action to perform")

    args = parser.parse_args()

    chroot_path = Path(os.getenv("CHROOT_PATH", "/opt/app"))
    pidfile_path = Path(os.getenv("PIDFILE_PATH", f"/opt/app/run/{args.daemon}.pid"))
    pidfile_path.parent.mkdir(parents=True, exist_ok=True)

    daemons = {
        "queuecleaner": QueueCleaner,
        "zmqdaemon": ZMQDaemon,
    }
    if args.daemon not in daemons:
        raise ValueError(f"Unknown daemon: {args.daemon}")
    daemon = daemons[args.daemon](pidfile=pidfile_path, chroot=chroot_path)
    if not hasattr(daemon, args.action):
        raise ValueError(f"Unknown action: {args.action}")
    action = getattr(daemon, args.action)()
