import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv

from pytorch_wavelets import DWTForward

class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        # 使用 Haar 小波进行一次小波变换
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        # 1x1 卷积用于通道融合（将4 * in_ch融合为out_ch）
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 小波分解：yL是低频部分，yH[0]是高频的三个方向
        yL, yH = self.wt(x)  # yL: (B, C, H/2, W/2), yH[0]: (B, C, 3, H/2, W/2)
        y_HL = yH[0][:, :, 0, :, :]  # HL方向（垂直边缘）
        y_LH = yH[0][:, :, 1, :, :]  # LH方向（水平边缘）
        y_HH = yH[0][:, :, 2, :, :]  # HH方向（对角边缘）
        # 通道拼接后进行卷积融合
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X


