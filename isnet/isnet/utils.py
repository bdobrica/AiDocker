import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

INPUT_WIDTH = 1024
INPUT_HEIGHT = 1024


def inference(model, img):
    h, w, _ = img.shape
    img_t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    img_t = torch.unsqueeze(img_t, 0)
    img_t = F.upsample(
        img_t, (INPUT_HEIGHT, INPUT_WIDTH), mode="bilinear"
    ).type(torch.uint8)
    inp_t = torch.divide(img_t, 255.0)
    inp_t = normalize(inp_t, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    mask_t = model(inp_t)
    mask_t = F.upsample(mask_t[0][0], (h, w), mode="bilinear")
    mask_t = torch.squeeze(mask_t, 0)

    mask_t_max = torch.max(mask_t)
    mask_t_min = torch.min(mask_t)

    mask_t = (mask_t - mask_t_min) / (mask_t_max - mask_t_min)
    mask_t = (mask_t * 255).permute(1, 2, 0)
    mask = mask_t.cpu().data.numpy()
    mask = mask.mean(axis=2).astype(np.uint8)

    return mask
