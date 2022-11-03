import numpy as np
import cv2
import torch

MAX_SIZE_H = 1600
MAX_SIZE_W = 1600


def gen_trimap_from_segmap_e2e(segmap):
    trimap = np.argmax(segmap, axis=1)[0]
    trimap = trimap.astype(np.int64)
    trimap[trimap == 1] = 128
    trimap[trimap == 2] = 255
    return trimap.astype(np.uint8)


def inference_img_scale(model, scale_img):
    tensor_img = torch.from_numpy(
        scale_img.astype(np.float32)[np.newaxis, :, :, :]
    ).permute(0, 3, 1, 2)

    input_t = tensor_img

    pred_global, pred_local, pred_fusion = model(input_t)
    pred_global = pred_global.data.cpu().numpy()
    pred_global = gen_trimap_from_segmap_e2e(pred_global)
    pred_local = pred_local.data.cpu().numpy()[0, 0, :, :]
    pred_fusion = pred_fusion.data.cpu().numpy()[0, 0, :, :]

    return pred_global, pred_local, pred_fusion


def get_masked_local_from_global_test(global_result, local_result):
    weighted_global = np.ones(global_result.shape)
    weighted_global[global_result == 255] = 0
    weighted_global[global_result == 0] = 0
    fusion_result = (
        global_result * (1.0 - weighted_global) / 255
        + local_result * weighted_global
    )
    return fusion_result


def inference(model, img):
    h, w, _ = img.shape
    new_h = min(MAX_SIZE_H, h - (h % 32))
    new_w = min(MAX_SIZE_W, w - (w % 32))

    global_ratio = 1 / 3
    local_ratio = 1 / 2
    resize_h = int(h * global_ratio)
    resize_w = int(w * global_ratio)
    new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
    new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
    scale_img = cv2.resize(img, (new_h, new_w)) * 255.0
    pred_glance_1, _, _ = inference_img_scale(model, scale_img)
    pred_glance_1 = cv2.resize(pred_glance_1, (h, w)) * 255.0
    resize_h = int(h * local_ratio)
    resize_w = int(w * local_ratio)
    new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
    new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
    scale_img = cv2.resize(img, (new_h, new_w)) * 255.0
    _, pred_focus_2, _ = inference_img_scale(model, scale_img)
    pred_focus_2 = cv2.resize(pred_focus_2, (h, w))
    pred_fusion = get_masked_local_from_global_test(pred_glance_1, pred_focus_2)

    return pred_glance_1, pred_focus_2, pred_fusion
