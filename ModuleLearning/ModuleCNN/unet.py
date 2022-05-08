import torch
import torch.nn as nn
from ModuleLearning.ModuleCNN.unet_helpers import DoubleConvDS,UpDS,DownDS,OutConv,CBAM,DoubleConv,Up, Down, DoubleDilatedConv, DownDilated, UpDilated, DoubleConv3d,Up3d, Down3d,OutConv3d, DoubleDilatedConv3d, DownDilated3d



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


        self.dconv_down1 = double_conv(dim_channels, 16, kernel_1 = (3,4), padding_1 = (1,1), kernel_2 = (3,4), padding_2 = (1,1)) # kernel1changed from 4,3 to 3,4 and kernel 2 from 4,4 to 3,4 when downsampled image #88x44
        self.dconv_down2 = double_conv(16, 32,  kernel_1 = (3,3), padding_1 = (1,1), kernel_2 = (3,3), padding_2 = (1,1)) #44*22
        self.dconv_down3 = double_conv(32, 64,  kernel_1 = (3,3), padding_1 = (1,1), kernel_2 = (3,3), padding_2 = (1,1)) #22*11
        #self.dconv_down4 = double_conv(256, 512,  kernel = (3,3), padding = (1,1))

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up2 = double_conv(32 + 64, 32, kernel_1 = (3,3), padding_1 = (1,1),kernel_2 = (3,3), padding_2 = (1,1))
        self.dconv_up1 = double_conv(32 + 16, 16,kernel_1 = (3,4), padding_1 = (1,2),kernel_2 = (3,4), padding_2 = (1,2)) #90*45 Kernel 1 (3,3) to (3,4) and padding_1 from 1,1 to 1,2 and padding 2 from 2,2, to 1,2

        self.conv_last = nn.Conv2d(16, last_channel_size, 1)


    def forward(self, x):
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
        #print("upsample", x.shape)
        x = torch.cat([x, conv1], dim=1)
        #print("concat",x.shape)
        x = self.dconv_up1(x)
        #print("up1", x.shape)
        out = self.conv_last(x)
        #print("conv last",out.shape)
        return out




class SmaAt_UNet_model(nn.Module):

    def __init__(self, dim_channels,last_channel_size=1, kernels_per_layer=2, bilinear=True, reduction_ratio=4): #16):

        super(SmaAt_UNet_model, self).__init__()
        self.n_channels = dim_channels
        self.n_classes = last_channel_size
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        factor = 2 if self.bilinear else 1

        self.inc = DoubleConvDS(self.n_channels, 16, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(16, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(16, 32, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(32, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(32, 64, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down4 = DownDS(128, (256) // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM((256) // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(256, (128) // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(128, (64) // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(64, (32) // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(32, 16, self.bilinear, kernels_per_layer=kernels_per_layer)


        self.outc = OutConv(16, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x1Att = self.cbam1(x1)
        # print(x1Att.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x2Att = self.cbam2(x2)
        # print(x2Att.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x3Att = self.cbam3(x3)
        # print(x3Att.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x4Att = self.cbam4(x4)
        # print(x4Att.shape)
        # x5 = self.down4(x4)
        # print(x5.shape)
        # x5Att = self.cbam5(x5)
        # print(x5Att.shape)
        x = self.up1(x4Att, x3Att)#self.up1(x5Att, x4Att)
        # print(x.shape)
        x = self.up2(x, x2Att)#self.up2(x, x3Att)
        # print(x.shape)
        x = self.up3(x, x1Att) #self.up3(x, x2Att)
        # print(x.shape)
        # x = self.up4(x, x1Att)
        # print(x.shape)
        #x = self.up1(x3Att, x2Att)
        # print(x.shape)
        #x = self.up2(x, x1Att)
        # print(x.shape)

        logits = self.outc(x)
        # print(x1.shape)
        return logits

class UNet_attn_model(nn.Module):

    def __init__(self, dim_channels,last_channel_size=1,bilinear=True, reduction_ratio=4): #16

        super(UNet_attn_model, self).__init__()
        self.n_channels = dim_channels
        self.n_classes = last_channel_size
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConv(self.n_channels, 16)
        self.cbam1 = CBAM(16, reduction_ratio=reduction_ratio)
        self.down1 = Down(16, 32)
        self.cbam2 = CBAM(32, reduction_ratio=reduction_ratio)
        self.down2 = Down(32, 64)
        self.cbam3 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down3 = Down(64, 128)
        self.cbam4 = CBAM(128, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        #self.down3 = Down(64, 128 // factor)
        #self.cbam4 = CBAM(128 // factor, reduction_ratio=reduction_ratio)
        self.down4 = Down(128, 256 // factor)
        self.cbam5 = CBAM(256 // factor, reduction_ratio=reduction_ratio)
        self.up1 = Up(256, 128 // factor, self.bilinear)#Up(128, 64 // factor, self.bilinear)#Up(256, 128 // factor, self.bilinear)
        self.up2 = Up(128, 64 // factor, self.bilinear) #Up(64, 32 // factor, self.bilinear)#Up(128, 64 // factor, self.bilinear)
        self.up3 = Up(64, 32 // factor, self.bilinear)#Up(32, 16, self.bilinear)#Up(64, 32 // factor, self.bilinear)
        self.up4 = Up(32, 16, self.bilinear)

        self.outc = OutConv(16, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)#self.up1(x4Att, x3Att)#self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)#self.up2(x, x2Att)#self.up2(x, x3Att)
        x = self.up3(x, x2Att) #self.up3(x, x1Att)#self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

class UNet_model(nn.Module):
    def __init__(self, dim_channels,last_channel_size=1,bilinear=True):
        super(UNet_model, self).__init__()
        self.n_channels = dim_channels
        self.n_classes = last_channel_size
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        #self.down3 = Down(64, 128)
        factor = 2 if self.bilinear else 1
        self.down3 = Down(64, 128 // factor)  #
        #self.down4 = Down(128, 256 // factor)
        self.up1 = Up(128, 64 // factor, self.bilinear) #Up(128, 64 // factor, self.bilinear) #Up(256, 128 // factor, self.bilinear)  #
        self.up2 = Up(64, 32 // factor, self.bilinear)  #Up(128, 64 // factor, self.bilinear) #Up(64, 32 // factor, self.bilinear) #
        self.up3 = Up(32, 16, self.bilinear) #Up(64, 32 // factor, self.bilinear) #Up(32, 16 , self.bilinear) #
        #self.up4 = Up(32, 16, self.bilinear)

        self.outc = OutConv(16, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        #x5 = self.down4(x4)
        # print(x5.shape)
        x = self.up1(x4, x3)  #self.up1(x5, x4) #self.up1(x4, x3) #
        # print(x.shape)
        x = self.up2(x, x2)  #self.up2(x, x3) #self.up2(x, x2) #
        # print(x.shape)
        x = self.up3(x, x1) #self.up3(x, x2) #self.up3(x, x1) #
        # print(x.shape)
        #x = self.up4(x, x1)
        # print(x.shape)
        logits = self.outc(x)
        # print(logits.shape)

        return logits

class UNet_model_ft(nn.Module):
    def __init__(self, pretrained_model, dim_channels, last_channel_size=1,input_channel_ft=16,bilinear=True):
        super(UNet_model_ft, self).__init__()
        self.n_channels = dim_channels
        self.n_classes = last_channel_size
        self.bilinear = bilinear
        self.input_channel_ft = input_channel_ft
        self.pretrained_model = pretrained_model

        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_model.model.up4.parameters():
            param.requires_grad=True
        for param in self.pretrained_model.model.up3.parameters():
            param.requires_grad=True

        #for param in self.pretrained_model.model.up2.parameters():
         #   param.requires_grad=True
        #for param in self.pretrained_model.model.up1.parameters():
         #   param.requires_grad=True
        #for param in self.pretrained_model.model.outc.parameters():
          #  param.requires_grad=True

        self.outc1 = DoubleConv(self.input_channel_ft, self.input_channel_ft) 

        self.outc2 = DoubleConv(self.input_channel_ft, self.input_channel_ft)
        self.final_outc = OutConv(self.input_channel_ft, self.n_classes)

    def forward(self, x):
        x1 = self.pretrained_model.model.inc(x)
        x2 = self.pretrained_model.model.down1(x1)
        x3 = self.pretrained_model.model.down2(x2)
        x4 = self.pretrained_model.model.down3(x3)
        x5 = self.pretrained_model.model.down4(x4)
        x = self.pretrained_model.model.up1(x5, x4)
        x = self.pretrained_model.model.up2(x, x3)
        x = self.pretrained_model.model.up3(x, x2)
        x = self.pretrained_model.model.up4(x, x1)
        x = self.outc1(x)
        x = self.outc2(x)
        logits = self.final_outc(x)
        #logits = self.pretrained_model.model.outc(x)
        return logits

class UNet3d_model(nn.Module):
    def __init__(self, dim_channels,last_channel_size=1,bilinear=True):
        super(UNet3d_model, self).__init__()
        self.n_channels = dim_channels
        self.n_classes = last_channel_size
        self.bilinear = bilinear

        self.inc = DoubleConv3d(self.n_channels, 8)
        self.down1 = Down3d(8, 16)
        self.down2 = Down3d(16, 32)
        self.down3 = Down3d(32, 64)
        factor = 2 if self.bilinear else 1
        self.down4 = Down3d(64, 128 // factor)
        self.up1 = Up3d(128, 64 // factor, self.bilinear)
        self.up2 = Up3d(64, 32 // factor, self.bilinear)
        self.up3 = Up3d(32, 16 // factor, self.bilinear)
        self.up4 = Up3d(16, 8, self.bilinear)

        self.outc = OutConv3d(8, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits


class Dilated_UNet3d_model(nn.Module):
    def __init__(self, dim_channels,last_channel_size=1,bilinear=True):
        super(Dilated_UNet3d_model, self).__init__()
        self.n_channels = dim_channels
        self.n_classes = last_channel_size
        self.bilinear = bilinear

        self.inc = DoubleDilatedConv3d(self.n_channels, 8)
        self.down1 = DownDilated3d(8, 16)
        self.down2 = DownDilated3d(16, 32)
        self.down3 = DownDilated3d(32, 64)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDilated3d(64, 128 // factor)
        self.bottleneck1 = DoubleDilatedConv3d(128 // factor, 128 // factor)
        self.up1 = Up3d(128, 64 // factor, self.bilinear)
        self.up2 = Up3d(64, 32 // factor, self.bilinear)
        self.up3 = Up3d(32, 16 // factor, self.bilinear)
        self.up4 = Up3d(16, 8, self.bilinear)

        self.outc = OutConv3d(8, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        print(x1.shape)
        x2 = self.down1(x1)
        print(x2.shape)
        x3 = self.down2(x2)
        print(x3.shape)
        x4 = self.down3(x3)
        print(x4.shape)
        x5 = self.down4(x4)
        print(x5.shape)
        x5 = self.bottleneck1(x5)
        print(x5.shape)
        x = self.up1(x5, x4)
        print(x.shape)
        x = self.up2(x, x3)
        print(x.shape)
        x = self.up3(x, x2)
        print(x.shape)
        x = self.up4(x, x1)
        print(x.shape)
        logits = self.outc(x)
        print(logits.shape)

        return logits


class Dilated_UNet_model(nn.Module):
    def __init__(self, dim_channels,last_channel_size=1,bilinear=True):
        super(Dilated_UNet_model, self).__init__()
        self.n_channels = dim_channels
        self.n_classes = last_channel_size
        self.bilinear = bilinear

        self.inc = DoubleDilatedConv(self.n_channels, 16)
        self.down1 = DownDilated(16, 32)
        self.down2 = DownDilated(32, 64)
        self.down3 = DownDilated(64, 128)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDilated(128, 256 // factor)
        self.bottleneck1 = DoubleDilatedConv(256 // factor, 256 // factor)
        # self.bottleneck2 = DoubleDilatedConv(256 // factor, 256 // factor, dilation1=4,double=False)

        self.up1 = Up(256, 128 // factor, self.bilinear)
        self.up2 = Up(128, 64 // factor, self.bilinear)
        self.up3 = Up(64, 32 // factor, self.bilinear)
        self.up4 = Up(32, 16, self.bilinear)

        self.outc = OutConv(16, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottleneck1(x5)
        # x5 = self.bottleneck2(x5)
        # print(x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Dilated_UNet_attn_model(nn.Module):
    def __init__(self, dim_channels,last_channel_size=1,bilinear=True, reduction_ratio=4):#16
        super(Dilated_UNet_attn_model, self).__init__()
        self.n_channels = dim_channels
        self.n_classes = last_channel_size
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleDilatedConv(self.n_channels, 16)
        self.cbam1 = CBAM(16, reduction_ratio=reduction_ratio)
        self.down1 = DownDilated(16, 32)
        self.cbam2 = CBAM(32, reduction_ratio=reduction_ratio)
        self.down2 = DownDilated(32, 64)
        self.cbam3 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down3 = DownDilated(64, 128)
        self.cbam4 = CBAM(128, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDilated(128, 256 // factor)
        self.cbam5 = CBAM(256 // factor, reduction_ratio=reduction_ratio)
        self.bottleneck1 = DoubleDilatedConv(256 // factor, 256 // factor)
        #self.bottleneck2 = DoubleDilatedConv(256 // factor, 256 // factor, dilation1=4, double=False)
        self.up1 = Up(256, 128 // factor, self.bilinear)
        self.up2 = Up(128, 64 // factor, self.bilinear)
        self.up3 = Up(64, 32 // factor, self.bilinear)
        self.up4 = Up(32, 16, self.bilinear)

        self.outc = OutConv(16, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5 = self.bottleneck1(x5)
        #x5 = self.bottleneck2(x5)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits





