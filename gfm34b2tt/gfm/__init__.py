"""
Bridging Composite and Real: Towards End-to-end Deep Image Matting [IJCV-2021]
Main network file (GFM).

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/GFM
Paper link (Arxiv): https://arxiv.org/abs/2010.16188

"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

SHORTER_PATH_LIMITATION = 1080


def collaborative_matting(glance_sigmoid, focus_sigmoid):
    values, index = torch.max(glance_sigmoid, 1)
    index = index[:, None, :, :].float()
    ### index <===> [0, 1, 2]
    ### bg_mask <===> [1, 0, 0]
    bg_mask = index.clone()
    bg_mask[bg_mask == 2] = 1
    bg_mask = 1 - bg_mask
    ### trimap_mask <===> [0, 1, 0]
    trimap_mask = index.clone()
    trimap_mask[trimap_mask == 2] = 0
    ### fg_mask <===> [0, 0, 1]
    fg_mask = index.clone()
    fg_mask[fg_mask == 1] = 0
    fg_mask[fg_mask == 2] = 1
    focus_sigmoid = focus_sigmoid.cpu()
    trimap_mask = trimap_mask.cpu()
    fg_mask = fg_mask.cpu()
    fusion_sigmoid = focus_sigmoid * trimap_mask + fg_mask
    fusion_sigmoid = fusion_sigmoid.cpu()
    return fusion_sigmoid


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv_up_psp(in_channels, out_channels, up_sample):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=up_sample, mode="bilinear", align_corners=False),
    )


def build_bb(in_channels, mid_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, 3, dilation=2, padding=2),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, 3, dilation=2, padding=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, dilation=2, padding=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def build_decoder(
    in_channels,
    mid_channels_1,
    mid_channels_2,
    out_channels,
    last_bnrelu,
    upsample_flag,
):
    layers = []
    layers += [
        nn.Conv2d(in_channels, mid_channels_1, 3, padding=1),
        nn.BatchNorm2d(mid_channels_1),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels_1, mid_channels_2, 3, padding=1),
        nn.BatchNorm2d(mid_channels_2),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels_2, out_channels, 3, padding=1),
    ]

    if last_bnrelu:
        layers += [
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

    if upsample_flag:
        layers += [nn.Upsample(scale_factor=2, mode="bilinear")]

    sequential = nn.Sequential(*layers)
    return sequential


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(
                input=stage(feats),
                size=(h, w),
                mode="bilinear",
                align_corners=True,
            )
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GFM(nn.Module):
    def __init__(self):
        super().__init__()

        self.gd_channel = 3

        RESNET34_PATH = os.getenv(
            "MODEL_PATH",
            "/opt/app/resnet34-b627a593.pth",
        )
        self.resnet = models.resnet34(weights=None)
        chkp = torch.load(RESNET34_PATH)
        self.resnet.load_state_dict(chkp)

        self.encoder0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4
        self.encoder5 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
        )
        self.encoder6 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
        )

        self.psp_module = PSPModule(512, 512, (1, 3, 5))
        self.psp6 = conv_up_psp(512, 512, 2)
        self.psp5 = conv_up_psp(512, 512, 4)
        self.psp4 = conv_up_psp(512, 256, 8)
        self.psp3 = conv_up_psp(512, 128, 16)
        self.psp2 = conv_up_psp(512, 64, 32)
        self.psp1 = conv_up_psp(512, 64, 32)
        self.decoder6_g = build_decoder(1024, 512, 512, 512, True, True)
        self.decoder5_g = build_decoder(1024, 512, 512, 512, True, True)
        self.decoder4_g = build_decoder(1024, 512, 512, 256, True, True)
        self.decoder3_g = build_decoder(512, 256, 256, 128, True, True)
        self.decoder2_g = build_decoder(256, 128, 128, 64, True, True)
        self.decoder1_g = build_decoder(128, 64, 64, 64, True, False)

        self.bridge_block = build_bb(512, 512, 512)
        self.decoder6_f = build_decoder(1024, 512, 512, 512, True, True)
        self.decoder5_f = build_decoder(1024, 512, 512, 512, True, True)
        self.decoder4_f = build_decoder(1024, 512, 512, 256, True, True)
        self.decoder3_f = build_decoder(512, 256, 256, 128, True, True)
        self.decoder2_f = build_decoder(256, 128, 128, 64, True, True)
        self.decoder1_f = build_decoder(128, 64, 64, 64, True, False)

        self.decoder0_g = nn.Sequential(nn.Conv2d(64, self.gd_channel, 3, padding=1))
        self.decoder0_f = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1))

    def forward(self, input):
        glance_sigmoid = torch.zeros(input.shape)
        focus_sigmoid = torch.zeros(input.shape)
        fusion_sigmoid = torch.zeros(input.shape)

        e0 = self.encoder0(input)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        psp = self.psp_module(e6)
        d6_g = self.decoder6_g(torch.cat((psp, e6), 1))
        d5_g = self.decoder5_g(torch.cat((self.psp6(psp), d6_g), 1))
        d4_g = self.decoder4_g(torch.cat((self.psp5(psp), d5_g), 1))

        d3_g = self.decoder3_g(torch.cat((self.psp4(psp), d4_g), 1))
        d2_g = self.decoder2_g(torch.cat((self.psp3(psp), d3_g), 1))
        d1_g = self.decoder1_g(torch.cat((self.psp2(psp), d2_g), 1))

        d0_g = self.decoder0_g(d1_g)

        glance_sigmoid = torch.sigmoid(d0_g)

        bb = self.bridge_block(e6)
        d6_f = self.decoder6_f(torch.cat((bb, e6), 1))
        d5_f = self.decoder5_f(torch.cat((d6_f, e5), 1))
        d4_f = self.decoder4_f(torch.cat((d5_f, e4), 1))

        d3_f = self.decoder3_f(torch.cat((d4_f, e3), 1))
        d2_f = self.decoder2_f(torch.cat((d3_f, e2), 1))
        d1_f = self.decoder1_f(torch.cat((d2_f, e1), 1))

        d0_f = self.decoder0_f(d1_f)
        focus_sigmoid = torch.sigmoid(d0_f)

        fusion_sigmoid = collaborative_matting(glance_sigmoid, focus_sigmoid)
        return glance_sigmoid, focus_sigmoid, fusion_sigmoid
