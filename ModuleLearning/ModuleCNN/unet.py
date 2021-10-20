import torch
import torch.nn as nn



def double_conv(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        # nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel, padding=padding),
        # nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, dim_channels, last_channel_size):
        super().__init__()

        self.dconv_down1 = double_conv(dim_channels, 64, kernel = (3,3), padding = (1,0))
        self.dconv_down2 = double_conv(64, 128,  kernel = (3,3), padding = (1,1))
        self.dconv_down3 = double_conv(128, 256,  kernel = (3,3), padding = (1,1))
        # self.dconv_down4 = double_conv(256, 512,  kernel = (3,3), padding = (1,1))

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.dconv_up3 = double_conv(256 + 512, 256,kernel = (3,3), padding = (1,1))
        self.dconv_up2 = double_conv(128 + 256, 128, kernel = (3,3), padding = (1,1))
        self.dconv_up1 = double_conv(128 + 64, 64,kernel = (3,3), padding = (1,2))

        self.conv_last = nn.Conv2d(64, last_channel_size, 1)

    def forward(self, x):
        # conv1 = self.dconv_down1(x)
        # #print("down1",conv1.shape)
        # x = self.maxpool(conv1)
        # #print("maxpool",x.shape)
        # conv2 = self.dconv_down2(x)
        # #print("down2", conv2.shape)
        # x = self.maxpool(conv2)
        # #print("maxpool",x.shape)
        # conv3 = self.dconv_down3(x)
        # #print("down3", conv3.shape)
        # x = self.maxpool(conv3)
        # #print("maxpool",x.shape)
        # x = self.dconv_down4(x)
        # #print("down4",x.shape)
        # x = self.upsample(x)
        # #print("upsample", x.shape)
        # x = torch.cat([x, conv3], dim=1)
        # #print("concat",x.shape)
        # x = self.dconv_up3(x)
        # #print("up3", x.shape)
        # x = self.upsample(x)
        # #print("upsample", x.shape)
        # x = torch.cat([x, conv2], dim=1)
        # #print("concat",x.shape)
        # x = self.dconv_up2(x)
        # #print("up2", x.shape)
        # x = self.upsample(x)
        # #print("upsample", x.shape)
        # x = torch.cat([x, conv1], dim=1)
        # #print("concat",x.shape)
        # x = self.dconv_up1(x)
        # #print("up1", x.shape)
        # out = self.conv_last(x)
        # #print("conv last",out.shape)
        # return out
        conv1 = self.dconv_down1(x)
        # print("down1",conv1.shape)
        x = self.maxpool(conv1)
        # print("maxpool",x.shape)
        conv2 = self.dconv_down2(x)
        # print("down2", conv2.shape)
        x = self.maxpool(conv2)
        # print("maxpool",x.shape)
        x = self.dconv_down3(x)
        # print("down3", conv3.shape)
        x = self.upsample(x)
        # print("upsample", x.shape)
        x = torch.cat([x, conv2], dim=1)
        # print("concat",x.shape)
        x = self.dconv_up2(x)
        # print("up2", x.shape)
        x = self.upsample(x)
        # print("upsample", x.shape)
        x = torch.cat([x, conv1], dim=1)
        # print("concat",x.shape)
        x = self.dconv_up1(x)
        # print("up1", x.shape)
        out = self.conv_last(x)
        # print("conv last",out.shape)
        return out