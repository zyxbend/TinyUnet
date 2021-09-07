
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from modeling.edge_guidance_module import EdgeGuidanceModule
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, BatchNorm=None):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )


    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, BatchNorm):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, BatchNorm)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, BatchNorm):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        # self.conv = double_conv(in_ch, out_ch, BatchNorm)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            BatchNorm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            BatchNorm(out_ch)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

    def get_bn_before_relu(self):
        return self.conv[-1]

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, inc_channels, sync_bn=False):
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        super(UNet, self).__init__()
        self.inc_c = inc_channels
        self.inc = double_conv(n_channels, inc_channels, BatchNorm)
        self.down1 = down(inc_channels, inc_channels * 2, BatchNorm)
        self.down2 = down(inc_channels * 2, inc_channels * 4, BatchNorm)
        self.down3 = down(inc_channels * 4, inc_channels * 8, BatchNorm)
        self.down4 = down(inc_channels * 8, inc_channels * 8, BatchNorm)
        self.up1 = up(inc_channels * 16, inc_channels * 4, BatchNorm)
        self.up2 = up(inc_channels * 8, inc_channels * 2, BatchNorm)
        self.up3 = up(inc_channels * 4, inc_channels, BatchNorm)
        self.up4 = up(inc_channels * 2, inc_channels, BatchNorm)
        self.outc = outconv(inc_channels, n_classes)
        self.relu = nn.ReLU(inplace=True)
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

        x = self.outc(x)

        return torch.sigmoid(x)

    ######################

    def get_bn_before_relu(self):
        bn0 = self.inc.conv[4]
        bn1 = self.down1.mpconv[-1].conv[4]
        bn2 = self.down2.mpconv[-1].conv[4]
        bn3 = self.down3.mpconv[-1].conv[4]
        bn4 = self.down4.mpconv[-1].conv[4]
        bn5 = self.up1.get_bn_before_relu()
        bn6 = self.up2.get_bn_before_relu()
        bn7 = self.up3.get_bn_before_relu()
        bn8 = self.up4.get_bn_before_relu()
        # BNs = [bn0]
        # BNs = [bn8]
        # BNs = [bn4]
        BNs = [bn1, bn3]
        # BNs = [bn0,bn2,bn4,bn6,bn8]
        # BNs = [bn0,bn1,bn2,bn3,bn4,bn5,bn6,bn7,bn8]

        return BNs


    ##########################################具体的通道数需要修改
    def get_channel_num(self):
        # 返回Unet各层通道数
        # return [64, 128, 256, 512, 512, 256, 128, 64, 64]


        # return [self.inc_c]
        # return [self.inc_c]
        # return [self.inc_c * 8]
        return [128, 512]
        # return [self.inc_c,self.inc_c * 4, self.inc_c * 8,self.inc_c * 2, self.inc_c]
        # return [self.inc_c, self.inc_c * 2,self.inc_c * 4,self.inc_c * 8,self.inc_c * 8, self.inc_c * 4,self.inc_c * 2,self.inc_c,self.inc_c]



    ##########################################        get_feature()函数到底需不需要参数传入
    def extract_feature(self, x):
        feat0 = self.inc(x)
        feat1 = self.down1(feat0)
        feat2 = self.down2(feat1)
        feat3 = self.down3(feat2)
        feat4 = self.down4(feat3)
        feat5 = self.up1(feat4, feat3)
        feat6 = self.up2(feat5, feat2)
        feat7 = self.up3(feat6, feat1)
        feat8 = self.up4(feat7, feat0)
        x = self.outc(feat8)
        out = torch.sigmoid(x)
        # return [feat0], out
        # return [feat8], out
        # return [feat4], out
        return [feat1, feat3], out
        # return [feat0,feat2,feat4,feat6,feat8], out
        # return [feat0, feat1,feat2,feat3,feat4,feat5,feat6,feat7,feat8] ,out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_1x_lr_params(self):
        modules = [self]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
