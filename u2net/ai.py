#!/usr/bin/env python3
import sys
import os
import time
import signal
from pathlib import Path
from daemon import Daemon

from skimage import io
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image

from u2net import U2NET

class AIDaemon(Daemon):
    def ai(self, source_file, prepared_file):
        pid = os.fork()
        if pid != 0:
            return

        try:
            so_data = SalObjDataset(
                img_name_list = [str(source_file)],
                lbl_name_list = [],
                transform = transforms.Compose([
                    RescaleT(320),
                    ToTensorLab(flag=0)]
                ))
            so_loader = DataLoader(
                so_data,
                batch_size = 1,
                shuffle = False,
                num_workers = 1
                )
            net = U2NET(3,1)
            MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/app/u2net.pth")
            net.load_state_dict(torch.load(MODEL_PATH, map_location = 'cpu'))
            net.eval()

            input_image = io.imread(str(source_file))
            input_label = np.zeros(input_image.shape[:2] + (1,), dtype=np.uint8)

            def rescale_t(image, output_size = 320):
                image_ = transform.resize(image, (output_size, output_size), mode='constant')
                return image_

            def to_tensor(image, flag = 0):
                image_ = image / np.max(image)

                image_[:,:,0] = (image_[:,:,0]-0.485)/0.229
                image_[:,:,1] = (image_[:,:,1]-0.456)/0.224
                image_[:,:,2] = (image_[:,:,2]-0.406)/0.225

                image_ = image_.transpose((2, 0, 1))
                
                return torch.from_numpy(image_)

            input_data = Variable(transforms.Compose([
                rescale_t,
                to_tensor_lab
            ])(input_image).type(torch.FloatTensor))

            output_data = net(input_data)
            pred = output_data[0][:,0,:,:]
            ma = torch.max(pred)
            mi = torch.min(pred)
            pred = (pred-mi)/(ma-mi)
            pred_np = pred.squeeze().cpu().data.numpy()
            pred_im = Image.fromarray(pred_np * 255).convert('RGB')
            output_image = pred_im.resize(input_image.shape[1::-1], resample = Image.BILINEAR)
            output_image.save(str(prepared_file))
        except Exception as e:
            pass
        
        source_file.unlink()
        sys.exit()

    def queue(self):
        STAGED_PATH = os.environ.get("STAGED_PATH", "/tmp/ai/staged")
        SOURCE_PATH = os.environ.get("SOURCE_PATH", "/tmp/ai/source")
        PREPARED_PATH = os.environ.get("PREPARED_PATH", "/tmp/ai/prepared")
        MAX_FORK = int(os.environ.get("MAX_FORK", 8))
        CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 4096))

        staged_files = { f for f in Path(STAGED_PATH).glob("*") if f.is_file() }
        source_files = { f for f in Path(SOURCE_PATH).glob("*") if f.is_file() }
        source_files_count = len(source_files)

        while source_files_count < MAX_FORK and staged_files:
            source_files_count += 1
            staged_file = staged_files.pop()
            source_file = Path(SOURCE_PATH) / staged_file.name
            prepared_file = Path(PREPARED_PATH) / (staged_file.stem + '.png')

            with staged_file.open('rb') as src_fp, source_file.open('wb') as dst_fp:
                while True:
                    chunk = src_fp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    dst_fp.write(chunk)

            staged_file.unlink()
            self.ai(source_file, prepared_file)
    
    def run(self):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        while True:
            self.queue()
            time.sleep(1.0)
            
if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AIDaemon(pidfile = PIDFILE_PATH, chroot = CHROOT_PATH).start()