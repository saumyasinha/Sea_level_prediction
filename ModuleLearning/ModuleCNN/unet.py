import torch
import torch.nn as nn



def double_conv(in_channels, out_channels, kernel_1, padding_1, kernel_2, padding_2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_1, padding=padding_1),
        # nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_2, padding=padding_2),
        # nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, dim_channels, last_channel_size):
        super().__init__()

        #
     #   self.dconv_down1 = double_conv(dim_channels, 16, kernel_1 = (4,3), padding_1 = (1,1), kernel_2 = (4,4), padding_2 = (1,1)) #88x44
     #   self.dconv_down2 = double_conv(16, 32,  kernel_1 = (3,3), padding_1 = (1,1), kernel_2 = (3,3), padding_2 = (1,1)) #44*22
     #   self.dconv_down3 = double_conv(32, 64,  kernel_1 = (3,3), padding_1 = (1,1), kernel_2 = (3,3), padding_2 = (1,1)) #22*11
        # #self.dconv_down4 = double_conv(256, 512,  kernel = (3,3), padding = (1,1))
        #
    #    self.maxpool = nn.MaxPool2d(2)
    #    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #
        # #self.dconv_up3 = double_conv(256 + 512, 256,kernel = (3,3), padding = (1,1))
        #
     #   self.dconv_up2 = double_conv(32 + 64, 32, kernel_1 = (3,3), padding_1 = (1,1),kernel_2 = (3,3), padding_2 = (1,1))
      #  self.dconv_up1 = double_conv(32 + 16, 16,kernel_1 = (3,3), padding_1 = (1,1),kernel_2 = (3,4), padding_2 = (2,2)) #90*45

      #  self.conv_last = nn.Conv2d(16, last_channel_size, 1)

        self.dconv_down1 = double_conv(dim_channels, 16, kernel_1=(5, 3), kernel_2=(5, 3), padding_1=(0, 0), padding_2=(0, 0))
        self.dconv_down2 = double_conv(16, 32, kernel_1=(3, 3), kernel_2=(3, 3), padding_1=(1, 1), padding_2=(1, 1))
        self.dconv_down3 = double_conv(32, 64, kernel_1=(3, 3), kernel_2=(3, 3), padding_1=(1, 1), padding_2=(1, 1))
       # # self.dconv_down4 = double_conv(256, 512, kernel=(3, 3), padding=(1, 1))

        self.maxpool = nn.MaxPool2d(4)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

       # # self.dconv_up3 = double_conv(256 + 512, 256, kernel=(3, 3), padding=(1, 1))
        self.dconv_up2 = double_conv(32 + 64, 32, kernel_1=(3, 3), kernel_2=(3, 3), padding_1=(1, 1), padding_2=(1, 1))
        self.dconv_up1 = double_conv(16 + 32, 16, kernel_1=(3, 3), kernel_2=(3, 3), padding_1=(3, 2), padding_2=(3, 2))

        self.conv_last = nn.Conv2d(16, last_channel_size, 1)

        #self.dconv_down1 = double_conv(dim_channels, 16, kernel_1=(3, 3), kernel_2=(3, 3), padding_1=(1, 0),
        #                                padding_2=(1, 0))
        #self.dconv_down2 = double_conv(16, 32, kernel_1=(3, 3), kernel_2=(3, 3), padding_1=(1, 1), padding_2=(1, 1))
       # self.dconv_down3 = double_conv(32, 64, kernel_1=(3, 3), kernel_2=(3, 3), padding_1=(1, 1), padding_2=(1, 1))
        # # self.dconv_down4 = double_conv(256, 512, kernel=(3, 3), padding=(1, 1))
        #
       # self.maxpool = nn.MaxPool2d(2)
       # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #
        # # self.dconv_up3 = double_conv(256 + 512, 256, kernel=(3, 3), padding=(1, 1))
       # self.dconv_up2 = double_conv(32 + 64, 32, kernel_1=(3, 3), kernel_2=(3, 3), padding_1=(1, 1), padding_2=(1, 1))
       # self.dconv_up1 = double_conv(16 + 32, 16, kernel_1=(3, 3), kernel_2=(3, 3), padding_1=(1, 2), padding_2=(1, 2))
        #
       # self.conv_last = nn.Conv2d(16, last_channel_size, 1)

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
        #print("down1",conv1.shape)
        x = self.maxpool(conv1)
        #print("maxpool",x.shape)
        conv2 = self.dconv_down2(x)
        #print("down2", conv2.shape)
        x = self.maxpool(conv2)

        #print("maxpool",x.shape)
        x = self.dconv_down3(x)
        #print("down3", x.shape)
        #x = self.maxpool(conv3)
        #print("maxpool",x.shape)
        #x = self.dconv_down4(x)
        #print("down4",x.shape)
        x = self.upsample(x)
        #print("upsample", x.shape)
        #x = torch.cat([x, conv3], dim=1)
        #print("concat",x.shape)
        #x = self.dconv_up3(x)
        #print("up3", x.shape)
        #x = self.upsample(x)
        #print("upsample", x.shape)
        x = torch.cat([x, conv2], dim=1)
        #print("concat",x.shape)
        x = self.dconv_up2(x)
        #print("up2", x.shape)
        x = self.upsample(x)
       # print("upsample", x.shape)
        x = torch.cat([x, conv1], dim=1)
        #print("concat",x.shape)
        x = self.dconv_up1(x)
        #print("up1", x.shape)
        out = self.conv_last(x)
        #print("conv last",out.shape)
        return out
