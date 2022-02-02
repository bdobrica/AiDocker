#!/bin/bash
if [ ! $(command -v gdown) ]; then
    pip install gdown
fi
if [ ! -f "modnet/modnet_photographic_portrait_matting.onnx" ]; then
    # Credits: https://github.com/ZHKKKe/MODNet
    gdown https://drive.google.com/uc?id=1IxxExwrUe4_yQnlEx389tmQI8luX7z5m --output modnet/modnet_photographic_portrait_matting.onnx
fi
if [ ! -f "u2net/u2net.pth" ]; then
    # Credits: https://github.com/xuebinqin/U-2-Net
    gdown https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ --output u2net/u2net.pth
fi
if [ ! -f "yolo4/yolov4.weights" ]; then
    # Credits: https://github.com/kiyoshiiriemon/yolov4_darknet
    gdown https://drive.google.com/uc?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT --output yolo4/yolov4.weights
fi
docker build -f modnet/Dockerfile -t modnet .
docker build -f u2net/Dockerfile -t u2net .
docker build -f yolov4/Dockerfile -t yolov4 .