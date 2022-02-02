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

        staged_files = []
        lifetime = 1.1 * float(FILE_LIFETIME)
        for meta_file in Path(STAGED_PATH).glob('*.json'):
            with meta_file.open('r') as fp:
                try:
                    image_metadata = json.load(fp)
                except:
                    image_metadata = {}
        
            if float(image_metadata.get('upload_time', 0)) +\
                lifetime < time.time():
                staged_file = meta_file.with_suffix(
                    image_metadata.get('extension', '.jpg'))
                if staged_file.is_file():
                    staged_files.append(staged_file)

        staged_files += list(filter(
            lambda f:\
                f.is_file() and f.stat().st_mtime + lifetime < time.time(),
            Path(STAGED_PATH).glob('*')))

        for staged_file in set(staged_files):
            meta_file = staged_file.with_suffix('.json')
            source_file = Path(SOURCE_PATH) / staged_file.name
            prepared_file = Path(PREPARED_PATH) / staged_file.name
            json_file = prepared_file.with_suffix('.json')

            if meta_file.is_file():
                try:
                    meta_file.unlink()
                except:
                    pass
            if source_file.is_file():
                try:
                    source_file.unlink()
                except:
                    pass
            if prepared_file.is_file():
                try:
                    prepared_file.unlink()
                except:
                    pass
            if json_file.is_file():
                try:
                    json_file.unlink()
                except:
                    pass
            if staged_file.is_file():
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