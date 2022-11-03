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
if [ ! -f "agenet/EfficientNetB3_224_weights.11-3.44.hdf5" ]; then
    wget "https://ublo.ro/wp-content/mirror/agenet/EfficientNetB3_224_weights.11-3.44.hdf5" -O agenet/EfficientNetB3_224_weights.11-3.44.hdf5
fi
if [ ! -f "gfm34b2tt/gfm_r34_2b_tt.pth" ]; then
    wget "https://ublo.ro/wp-content/mirror/gfm/gfm_r34_2b_tt.pth" -O gfm34b2tt/gfm_r34_2b_tt.pth
fi
if [ ! -f "gfm34b2tt/resnet34-b627a593.pth" ]; then
    wget "https://ublo.ro/wp-content/mirror/pytorch/resnet34-b627a593.pth" -O gfm34b2tt/resnet34-b627a593.pth
fi
if [ ! -f "isnet/isnet-general-use.pth" ]; then
    wget "https://ublo.ro/wp-content/mirror/isnet/isnet-general-use.pth" -O isnet/isnet-general-use.pth
fi

docker build -f modnet/Dockerfile -t modnet .
docker build -f u2net/Dockerfile -t u2net .
docker build -f yolov4/Dockerfile -t yolov4 .
docker build -f nudenet/Dockerfile -t nudenet .
docker build -f agenet/Dockerfile -t agenet .
docker build -f gfm34b2tt/Dockerfile -t gfm34b2tt .
docker build -f isnet/Dockerfile -t isnet .
