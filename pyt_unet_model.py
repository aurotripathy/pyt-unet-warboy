import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, input_shape):
        super(UNet, self).__init__()

        depths = [16, 16, 32, 32, 64]

        self.down1 = nn.Sequential(
            nn.Conv2d(input_shape[1], depths[0], 3, padding=1),
            nn.BatchNorm2d(depths[0], affine=True),
            nn.ReLU(),
            nn.Conv2d(depths[0], depths[0], 3, padding=1),
            nn.BatchNorm2d(depths[0]),
            nn.ReLU(),
            # nn.MaxPool2d(2, stride=2)
        )
        self.mxpl1 = nn.MaxPool2d(2, stride=2)

        self.down2 = nn.Sequential(
            nn.Conv2d(depths[0], depths[1], 3, padding=1),
            nn.BatchNorm2d(depths[1]),
            nn.ReLU(),
            nn.Conv2d(depths[1], depths[1], 3, padding=1),
            nn.BatchNorm2d(depths[1]),
            nn.ReLU(),
            # nn.MaxPool2d(2, stride=2)
        )
        self.mxpl2 = nn.MaxPool2d(2, stride=2)

        self.down3 = nn.Sequential(
            nn.Conv2d(depths[1], depths[2], 3, padding=1),
            nn.BatchNorm2d(depths[2]),
            nn.ReLU(),
            nn.Conv2d(depths[2], depths[2], 3, padding=1),
            nn.BatchNorm2d(depths[2]),
            nn.ReLU(),
            # nn.MaxPool2d(2, stride=2)
        )
        self.mxpl3 = nn.MaxPool2d(2, stride=2)

        self.down4 = nn.Sequential(
            nn.Conv2d(depths[2], depths[3], 3, padding=1),
            nn.BatchNorm2d(depths[3]),
            nn.ReLU(),
            nn.Conv2d(depths[3], depths[3], 3, padding=1),
            nn.BatchNorm2d(depths[3]),
            nn.ReLU(),
            # nn.MaxPool2d(2, stride=2)
        )
        self.mxpl4 = nn.MaxPool2d(2, stride=2)

        self.center = nn.Sequential(
            nn.Conv2d(depths[3], depths[4], 3, padding=1),
            nn.BatchNorm2d(depths[4]),
            nn.ReLU(),
            nn.Conv2d(depths[4], depths[4], 3, padding=1),
            nn.BatchNorm2d(depths[4]),
            nn.ReLU()
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(depths[4] + depths[3], depths[3], 3, padding=1),
            nn.BatchNorm2d(depths[3]),
            nn.ReLU(),
            nn.Conv2d(depths[3], depths[3], 3, padding=1),
            nn.BatchNorm2d(depths[3]),
            nn.ReLU(),
            nn.Conv2d(depths[3], depths[3], 3, padding=1),
            nn.BatchNorm2d(depths[3]),
            nn.ReLU()
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(depths[3] + depths[2], depths[2], 3, padding=1),
            nn.BatchNorm2d(depths[2]),
            nn.ReLU(),
            nn.Conv2d(depths[2], depths[2], 3, padding=1),
            nn.BatchNorm2d(depths[2]),
            nn.ReLU(),
            nn.Conv2d(depths[2], depths[2], 3, padding=1),
            nn.BatchNorm2d(depths[2]),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(depths[2] + depths[1], depths[1], 3, padding=1),
            nn.BatchNorm2d(depths[1]),
            nn.ReLU(),
            nn.Conv2d(depths[1], depths[1], 3, padding=1),
            nn.BatchNorm2d(depths[1]),
            nn.ReLU(),
            nn.Conv2d(depths[1], depths[1], 3, padding=1),
            nn.BatchNorm2d(depths[1]),
            nn.ReLU()
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(depths[1] + depths[0], depths[0], 3, padding=1),
            nn.BatchNorm2d(depths[0]),
            nn.ReLU(),
            nn.Conv2d(depths[0], depths[0], 3, padding=1),
            nn.BatchNorm2d(depths[0]),
            nn.ReLU(),
            nn.Conv2d(depths[0], depths[0], 3, padding=1),
            nn.BatchNorm2d(depths[0]),
            nn.ReLU()
        )

        self.classify = nn.Conv2d(depths[0], 1, 1)

    def forward(self, x):
        down1 = self.down1(x)
        down1 = self.mxpl1(down1)
        
        down2 = self.down2(down1)
        down2 = self.mxpl2(down2)

        down3 = self.down3(down2)
        down3 = self.mxpl3(down3)


        down4 = self.down4(down3)
        down4 = self.mxpl4(down4)

        center = self.center(down4)
        up4 = self.up4(torch.cat((center, down4), dim=1))
        up3 = self.up3(torch.cat((up4, down3), dim=1))
        up2 = self.up2(torch.cat((up3, down2), dim=1))
        up1 = self.up1(torch.cat((up2, down1), dim=1))
        classify = self.classify(up1)
        return classify
