#!/usr/bin/env python3
import sys
import os
import time
import signal
import json
import random
from pathlib import Path
from daemon import Daemon

import torch
import torchvision
import numpy as np
import cv2
from yolov4 import Darknet, non_max_suppression, scale_coords

class AIDaemon(Daemon):
    def ai(self, source_file, prepared_file):
        pid = os.fork()
        if pid != 0:
            return

        try:
            MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/app/yolov4.weights")
            CLASSES_PATH = os.environ.get("CLASSES_PATH", "/opt/app/coco.names")
            IMAGE_SIZE = 640
            AUTO_SIZE = 64
            BORDER_COLOR = (114, 114, 114)

            # Initialize
            device = torch.device('cpu')

            # Load model
            model = Darknet(None, IMAGE_SIZE).cpu()
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model'])
            except:
                model.load_darknet_weights(MODEL_PATH)
            model.to(device).eval()

            # Get names and colors
            with Path(CLASSES_PATH).open('r') as f:
                names = list(filter(None, f.read().split('\n')))  # filter removes empty strings (such as last line)

            img_orig = cv2.imread(str(source_file))

            # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
            shape = img_orig.shape[:2]  # current shape [height, width]
            new_shape = (IMAGE_SIZE, IMAGE_SIZE)  # desired new shape [height, width]
            # Scale ratio (new / old)
            ratio_ = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
            # Resize the image with padded border
            ratio = ratio_, ratio_ # width, height ratios
            new_unpad = int(round(shape[1] * ratio_)), int(round(shape[0] * ratio_))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] # wh padding
            dw, dh = np.mod(dw, AUTO_SIZE), np.mod(dh, AUTO_SIZE) # wh padding
            dw /= 2 # divide padding into 2 sides
            dh /= 2
            if shape[::-1] != new_unpad:  # resize
                img = cv2.resize(img_orig, new_unpad, interpolation=cv2.INTER_LINEAR)
            else:
                img = img_orig.copy()
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BORDER_COLOR)  # add border

            # Convert the image to the expected format
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Inference
            pred = model(img, augment = False)[0]

            # Do non-maximum suppression
            pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5, classes=None, agnostic=True)

            results = []

            img_copy = img_orig.copy()
            for i, det in enumerate(pred):  # detections per image
                gn = torch.tensor(img_orig.shape)[[1, 0, 1, 0]]
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_orig.shape).round()

                    for *xyxy, conf, class_id in det:
                        det_x = float((xyxy[0] + xyxy[2]) / 2)  # x center
                        det_y = float((xyxy[1] + xyxy[3]) / 2)  # y center
                        det_w = float(xyxy[2] - xyxy[0])  # width
                        det_h = float(xyxy[3] - xyxy[1])  # height

                        results.append({
                            'class': names[int(class_id)],
                            'conf': float(conf),
                            'x': det_x,
                            'y': det_y,
                            'w': det_w,
                            'h': det_h,
                            'area': det_w * det_h / (img_orig.shape[0] * img_orig.shape[1]),
                        })

                        color = [random.randint(0, 255) for _ in range(3)]
                        cv2.rectangle(
                            img_copy,
                            (int(det_x - det_w / 2.0), int(det_y - det_h / 2.0)),
                            (int(det_x + det_w / 2.0), int(det_y + det_h / 2.0)),
                            color,
                            thickness=2)
                        
                        text_size = cv2.getTextSize(names[int(class_id)], 0, fontScale=0.5, thickness=1)[0]
                        cv2.rectangle(
                            img_copy,
                            (int(det_x - det_w / 2.0), int(det_y - det_h / 2.0)),
                            (int(det_x - det_w / 2.0 + text_size[0]), int(det_y - det_h / 2.0 - text_size[1] - 3)),
                            color,
                            -1)
                        cv2.putText(
                            img_copy,
                            names[int(class_id)],
                            (int(det_x - det_w / 2.0), int(det_y - det_h / 2.0 - 2)),
                            0,
                            fontScale = 0.5,
                            color = (255, 255, 255),
                            thickness = 1,
                            lineType = cv2.LINE_AA)

            results.sort(key = lambda x : x.get('area') or 0.0, reverse = True)

            with prepared_file.open('w') as f:
                json.dump({'results':results}, f)
            
            cv2.imwrite(str(prepared_file.parent / (prepared_file.stem + '.png')), img_copy)
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

        staged_files = sorted([ f for f in Path(STAGED_PATH).glob("*") if f.is_file() ], key = lambda f : f.stat().st_mtime)
        source_files = [ f for f in Path(SOURCE_PATH).glob("*") if f.is_file() ]
        source_files_count = len(source_files)

        while source_files_count < MAX_FORK and staged_files:
            source_files_count += 1
            staged_file = staged_files.pop(0)
            source_file = Path(SOURCE_PATH) / staged_file.name
            prepared_file = Path(PREPARED_PATH) / (staged_file.stem + '.json')

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