#!/usr/bin/env python3
import json
import os
from pathlib import Path
from urllib import request

import cv2
import numpy as np

import pixellib
from daemon import ImageDaemon
from pixellib.torchbackend.instance import instanceSegmentation

__version__ = "0.9.0"


class AIDaemon(ImageDaemon):
    def ai(self, source_file, prepared_file, **metadata):
        pass


if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AIDaemon(pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
