#!/usr/bin/env python3
import sys
import os
import time
import signal
import json
import random
from pathlib import Path
from daemon import Daemon

class Cleaner(Daemon):
    def clean(self):
        STAGED_PATH = os.environ.get("STAGED_PATH", "/tmp/ai/staged")
        SOURCE_PATH = os.environ.get("SOURCE_PATH", "/tmp/ai/source")
        PREPARED_PATH = os.environ.get("PREPARED_PATH", "/tmp/ai/prepared")
        FILE_LIFETIME = os.environ.get('API_CLEANER_FILE_LIFETIME', '1800.0')

        lifetime = 1.1 * float(FILE_LIFETIME)
        staged_files = filter(
            lambda f: f.is_file() and f.stat().st_mtime + lifetime > time.time(),
            Path(STAGED_PATH).glob('*'))

        for staged_file in staged_files:
            for source_file in Path(SOURCE_PATH).glob(staged_file.stem + '.*'):
                try:
                    source_file.unlink()
                except:
                    pass
            for prepared_file in Path(PREPARED_PATH).glob(staged_file.stem + '.*'):
                try:
                    prepared_file.unlink()
                except:
                    pass
            try:
                staged_file.unlink()
            except:
                pass

    def run(self):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        FILE_LIFETIME = os.environ.get('API_CLEANER_FILE_LIFETIME', '1800.0')
        sleep_interval = min(
            0.05 * float(FILE_LIFETIME),
            float(os.environ.get('API_CLEANER_INTERVAL', '5.0'))
        )
        while True:
            self.clean()
            time.sleep(sleep_interval)
            
if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/cleaner.pid")

    Cleaner(pidfile = PIDFILE_PATH, chroot = CHROOT_PATH).start()