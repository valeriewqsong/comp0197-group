# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim
import torchvision

from torchvision.models import resnet

class decoder_block(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_size, in_size//4, 1)
        self.norm1 = nn.BatchNorm2d(in_size//4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_size//4, in_size//4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_size//4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_size//4, out_size, 1)
        self.norm3 = nn.BatchNorm2d(out_size)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class link_net(nn.Module):
    def __init__(self, classes, encoder='resnet34'):
        super().__init__()

        # Encoder
        res = resnet.resnet34(weights='DEFAULT')
        
        self.conv = res.conv1
        self.bn = res.bn1
        self.relu = res.relu
        self.maxpool = res.maxpool
        self.encoder1 = res.layer1
        self.encoder2 = res.layer2
        self.encoder3 = res.layer3
        self.encoder4 = res.layer4

        # Decoder
        self.decoder4 = decoder_block(512, 256)
        self.decoder3 = decoder_block(256, 128)
        self.decoder2 = decoder_block(128, 64)
        self.decoder1 = decoder_block(64, 64)

        self.finaldeconv = nn.ConvTranspose2d(64, 32, 3,
                                               stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Classification
        x = self.finaldeconv(d1)
        x = self.finalrelu1(x)
        x = self.finalconv2(x)
        x = self.finalrelu2(x)
        x = self.finalconv3(x)
        return x
