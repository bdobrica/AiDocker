#!/usr/bin/env python3
import sys
import os
import time
import signal
from pathlib import Path
from daemon import Daemon
import cv2
import numpy as np
import onnx
import onnxruntime

class AIDaemon(Daemon):
    def ai(self, source_file, prepared_file):
        pid = os.fork()
        if pid != 0:
            return

        try:
            ref_size = 512

            im = cv2.imread(str(source_file))
            out_im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            if len(im.shape) == 2:
                im = im[:, :, None]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis = 2)
            elif im.shape[2] == 4:
                im = im[:, :, 0:3]

            im = (im - 127.5) / 127.5
            im_h, im_w, im_c = im.shape

            def get_scale_factor(im_h, im_w, ref_size):
                if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
                    if im_w >= im_h:
                        im_rh = ref_size
                        im_rw = int(im_w / im_h * ref_size)
                    elif im_w < im_h:
                        im_rw = ref_size
                        im_rh = int(im_h / im_w * ref_size)
                else:
                    im_rh = im_h
                    im_rw = im_w

                im_rw = im_rw - im_rw % 32
                im_rh = im_rh - im_rh % 32

                x_scale_factor = im_rw / im_w
                y_scale_factor = im_rh / im_h

                return x_scale_factor, y_scale_factor

            x, y = get_scale_factor(im_h, im_w, ref_size)

            im = cv2.resize(im, None, fx = x, fy = y, interpolation = cv2.INTER_AREA)
            im = np.transpose(im)
            im = np.swapaxes(im, 1, 2)
            im = np.expand_dims(im, axis = 0).astype('float32')

            MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/app/modnet_photographic_portrait_matting.onnx")
            session = onnxruntime.InferenceSession(MODEL_PATH, None)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            result = session.run([output_name], {input_name: im})

            matte = (np.squeeze(result[0]) * 255).astype('uint8')
            matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation = cv2.INTER_AREA)
            out_im[:, :, 3] = matte

            out_im = out_im.astype(float)
            out_im[:,:,:3] = out_im[:,:,:3] * np.repeat(out_im[:,:,3].reshape(out_im.shape[:2] + (1,)) / 255.0, 3, axis = 2)

            cv2.imwrite(str(prepared_file), out_im.astype('uint8'))
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
        signal.signal(signal.SIGCHLD, self.SIG_IGN)
        while True:
            self.queue()
            time.sleep(1.0)
            
if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AIDaemon(pidfile = PIDFILE_PATH, chroot = CHROOT_PATH).start()