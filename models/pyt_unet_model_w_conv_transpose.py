

import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, input_shape):
        super(UNet, self).__init__()

        depths=[16, 16, 32, 32, 64]
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down1 = self.contract_block(input_shape[1], depths[0])
        self.down2 = self.contract_block(depths[0], depths[1])
        self.down3 = self.contract_block(depths[1], depths[2])
        self.down4 = self.contract_block(depths[2], depths[3])

        self.mid = nn.Sequential(
            nn.Conv2d(depths[3], depths[4], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(depths[4], depths[4], kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = self.expand_block(depths[4], depths[3])
        self.up2 = self.expand_block(depths[3]*2, depths[2])
        self.up3 = self.expand_block(depths[2]*2, depths[1])
        self.up4 = self.expand_block(depths[1]*2, depths[0])

        self.final = nn.Conv2d(depths[0]*2, 1, kernel_size=1)

    def contract_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(),
        )
        return block

    def expand_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),            
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        return block

    def forward(self, x):
        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        x = self.maxpool(conv3)

        conv4 = self.down4(x)
        x = self.maxpool(conv4)

        x = self.mid(x)

        x = self.up1(x)
        x = torch.cat([x, conv4], dim=1)

        x = self.up2(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.up3(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.up4(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.final(x)

        return x
