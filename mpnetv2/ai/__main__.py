"""
The main entry point of the AI module, which starts the AI daemon(s). There are 2 daemons:
- AiBuildDaemon: Reads documents as input and stores them in Redis. The documents are split into text items, which are
    stored as Redis hashes. The process is asynchronous.
- AiQueryDaemon: Reads text items as input and searches for similar text items in Redis. The process is synchronous.

Usage:
```bash
# Start the build daemon
python -m mpnetv2.ai --daemon build

# Start the query daemon
python -m mpnetv2.ai --daemon query
```
"""

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
    PIDFILE_PATH = os.getenv("PIDFILE_PATH", f"/opt/app/run/{args.daemon.lower()}.pid")

    if args.daemon.lower() in ("aibuilddaemon", "builddaemon", "build"):
        AiBuildDaemon(input_type=AiBuildBatch, pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
    elif args.daemon.lower() in ("aiquerydaemon", "querydaemon", "query"):
        AiQueryDaemon(input_type=AiQueryInput, pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
    else:
        raise ValueError(f"Unknown daemon type: {args.daemon}")
