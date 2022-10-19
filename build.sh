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
if [ ! -f "yolov4/coco.names" ]; then
    # Credits: https://github.com/kiyoshiiriemon/yolov4_darknet
    wget https://raw.githubusercontent.com/kiyoshiiriemon/yolov4_darknet/master/data/coco.names -O yolov4/coco.names
fi
if [ ! -f "yolov4/yolov4.weights" ]; then
    gdown https://drive.google.com/uc?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT --output yolov4/yolov4.weights
fi
NUDENET_FILES=(
    "nudenet/detector_v2_default_checkpoint.onnx"\
    "nudenet/detector_v2_default_classes"\
    "nudenet/detector_v2_base_checkpoint.onnx"\
    "nudenet/detector_v2_base_classes"\
)
for nudenet_file in "${NUDENET_FILES[@]}"; do
    if [ ! -f "${nudenet_file}" ]; then
        wget "https://ublo.ro/wp-content/mirror/${nudenet_file}" -O ${nudenet_file}
    fi
done

docker build -f modnet/Dockerfile -t modnet .
docker build -f u2net/Dockerfile -t u2net .
docker build -f yolov4/Dockerfile -t yolov4 .
docker build -f nudenet/Dockerfile -t nudenet .
