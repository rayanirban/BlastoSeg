from pydoc import stripid
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, (1,3,3), padding=(0,1,1), stride=(1,1,1)),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, (1,3,3), padding=(0,1,1), stride=(1,1,1)),
        nn.ReLU(inplace=True)
    )
def double_Tconv_up(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, (1,4,4), padding=(0,1,1), stride=(1,2,2)),
        nn.ReLU(inplace=True)
    )
class Unet3D(torch.nn.Module):

    def __init__(self, n_classes):
        super(Unet3D, self).__init__()

        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.convT1 = double_Tconv_up(512,512)
        self.convT2 = double_Tconv_up(256,256)
        self.convT3 = double_Tconv_up(128,128)
        self.convT4 = double_Tconv_up(64,64)


        self.maxpool = nn.MaxPool3d(kernel_size=(1,2,2))
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv3d(64, 1, 1)

    def forward(self, img):

        conv1 = self.dconv_down1(img)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        #x = self.upsample(x)
        x = self.convT1(x) 
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        #x = self.upsample(x)
        x = self.convT2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        #x = self.upsample(x)
        x = self.convT3(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)

        return out