import numpy as np
import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride),
                                  nn.BatchNorm2d(out_ch),
                                  nn.Dropout2d(p=0.2, inplace=True),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class LinearBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.4):
        super(LinearBlock, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_ch, out_ch),
                                  nn.Dropout(dropout_rate),
                                  nn.ReLU())

    def forward(self, x):
        return self.linear(x)


class ConditionalNet(nn.Module):
    def __init__(self):
        super(ConditionalNet, self).__init__()
        self.image_block = nn.Sequential(
            ConvBlock(3, 32, kernel_size=5, stride=2),
            ConvBlock(32, 32, kernel_size=3, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 64, kernel_size=3, stride=1),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ConvBlock(128, 128, kernel_size=3, stride=1),
            ConvBlock(128, 256, kernel_size=3, stride=1),
            ConvBlock(256, 256, kernel_size=3, stride=1)
        )

        self.image_fc = nn.Sequential(
            # nn.Linear(8192, 512),
            # nn.Dropout(0.5, inplace=True),
            # nn.ReLU(),
            LinearBlock(8192, 512, 0.3),
            LinearBlock(512, 512, 0.3)
            # nn.Linear(512, 512),
            # nn.Dropout(0.5, inplace=True),
            # nn.ReLU()
        )

        self.measure_fc = nn.Sequential(
            LinearBlock(1, 128),
            LinearBlock(128, 128)
            # nn.Linear(1, 128),
            # nn.Dropout(0.5, inplace=True),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.Dropout(0.5, inplace=True),
            # nn.ReLU()
        )

        self.embedding = nn.Sequential(
            LinearBlock(640, 512)
            # nn.Linear(640, 512),
            # nn.Dropout(0.5, inplace=True),
            # nn.ReLU
        )

        self.branches = nn.ModuleList([nn.Sequential(
            LinearBlock(512, 256),
            # nn.Linear(256, 256),
            LinearBlock(256,256),
            # nn.ReLU(),
            nn.Linear(256, 3)
        )
            for i in range(4)])

        self.pred_speed_branch = nn.Sequential(
            LinearBlock(512, 256),
            # nn.Linear(256, 256),
            LinearBlock(256,256),
            # nn.ReLU(),
            nn.Linear(256, 1)
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.1)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, img, speed):
        img = self.image_block(img)
        img = img.reshape(-1, 8192)
        img = self.image_fc(img)

        speed = self.measure_fc(speed)
        concat = torch.cat([img, speed], dim=1)
        concat = self.embedding(concat)
        res_matrix = torch.cat([branch(concat) for branch in self.branches], dim=1)
        pred_speed = self.pred_speed_branch(img)

        return res_matrix, pred_speed




