#!/usr/bin/env python3
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from urllib import request

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms

from daemon import ImageDaemon
from u2net import U2NET

__version__ = "0.9.0"


class AIDaemon(ImageDaemon):
    def ai(self, source_file, prepared_file, **metadata):
        pid = os.fork()
        if pid != 0:
            return

        try:
            net = U2NET(3, 1)
            MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/app/u2net.pth")
            net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            net.eval()

            im = cv2.imread(str(source_file))
            out_im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
            im = im[:, :, ::-1]

            def rescale_t(image, output_size=320):
                image_ = cv2.resize(
                    image,
                    dsize=(output_size, output_size),
                    interpolation=cv2.INTER_AREA,
                )
                return image_

            def to_tensor_lab(image, flag=0):
                image_ = image / np.max(image)

                image_[:, :, 0] = (image_[:, :, 0] - 0.485) / 0.229
                image_[:, :, 1] = (image_[:, :, 1] - 0.456) / 0.224
                image_[:, :, 2] = (image_[:, :, 2] - 0.406) / 0.225

                image_ = image_.transpose((2, 0, 1))
                image_ = image_[np.newaxis, :, :, :]

                return torch.from_numpy(image_)

            input_data = Variable(
                transforms.Compose([rescale_t, to_tensor_lab])(im).type(
                    torch.FloatTensor
                )
            )

            output_data = net(input_data)
            pred = output_data[0][:, 0, :, :]
            ma = torch.max(pred)
            mi = torch.min(pred)
            pred = (pred - mi) / (ma - mi)
            pred_np = pred.squeeze().cpu().data.numpy()

            sal_map = (pred_np * 255).astype("uint8")
            sal_map = cv2.resize(
                sal_map, im.shape[1::-1], interpolation=cv2.INTER_AREA
            )
            out_im[:, :, 3] = sal_map
            out_im = out_im.astype(float)

            im_h, im_w = out_im.shape[:2]

            background = metadata.get("background", "")
            color_re = re.compile(r"(^[A-Za-z0-9]{6}$)|(^[A-Za-z0-9]{8}$)")
            background_alpha = 0
            background_im = None
            if background.startswith("http://") or background.startswith(
                "https://"
            ):
                try:
                    req = request.urlopen(background)
                    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                    background_im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                    background_im = cv2.resize(
                        background_im,
                        dsize=(im_w, im_h),
                        interpolation=cv2.INTER_AREA,
                    )
                    if background_im.shape[2] == 4:
                        background_im = background_im.astype(float)
                        background_im = background_im[:, :, :3] * np.repeat(
                            background_im[:, :, 3].reshape(
                                out_im.shape[:2] + (1,)
                            )
                            / 255.0,
                            3,
                            axis=2,
                        )
                    background_alpha = 1.0
                except:
                    background_im = None
            elif color_re.match(background):
                red = int(background[0:2], 16)
                green = int(background[2:4], 16)
                blue = int(background[4:6], 16)
                if len(background) == 8:
                    background_alpha = int(background[6:8], 16) / 255.0
                else:
                    background_alpha = 1.0
                background_im = np.full(
                    (im_h, im_w, 3), [blue, green, red], dtype=np.float32
                )

            alpha_mask = np.repeat(
                out_im[:, :, 3].reshape(out_im.shape[:2] + (1,)) / 255.0,
                3,
                axis=2,
            )
            out_im[:, :, :3] = out_im[:, :, :3] * alpha_mask

            if (
                background_im is None
                and metadata.get("type", "") != "image/png"
            ):
                background_im = np.full(
                    (im_h, im_w, 3), [255.0, 255.0, 255.0], dtype=np.float32
                )
                background_alpha = 1.0

            if background_im is not None:
                out_im[:, :, :3] = out_im[:, :, :3] + background_im[
                    :, :, :3
                ] * background_alpha * (1.0 - alpha_mask)
                out_im[:, :, 3] = 255.0

            cv2.imwrite(str(prepared_file), out_im.astype("uint8"))
        except Exception as e:
            pass

        source_file.unlink()
        sys.exit()


if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AIDaemon(pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
