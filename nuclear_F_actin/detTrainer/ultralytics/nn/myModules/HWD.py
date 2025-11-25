# from pytorch_wavelets import DWTForward
import torch
import torch.nn as nn
import torch.nn.functional as F

class DWT(nn.Module):
    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        self.wave = wave
        self.mode = mode

        if wave == 'db1' or wave == 'haar':
            h0 = [0.7071067811865476, 0.7071067811865476]
            h1 = [-0.7071067811865476, 0.7071067811865476]
        elif wave == 'db4':
            h0 = [
                -0.010597401784997278, 0.032883011666982945,
                0.030841381835986965, -0.18703481171888114,
                -0.02798376941698385, 0.6308807679295904,
                0.7148465705525415, 0.23037781330885523
            ]
            h1 = [
                -0.23037781330885523, 0.7148465705525415,
                -0.6308807679295904, -0.02798376941698385,
                0.18703481171888114, 0.030841381835986965,
                -0.032883011666982945, -0.010597401784997278
            ]
        else:
            h0 = [0.7071067811865476, 0.7071067811865476]
            h1 = [-0.7071067811865476, 0.7071067811865476]

        self.register_buffer('h0_h', torch.tensor(h0, dtype=torch.float32).view(1, 1, 1, -1))
        self.register_buffer('h1_h', torch.tensor(h1, dtype=torch.float32).view(1, 1, 1, -1))

        self.register_buffer('h0_v', torch.tensor(h0, dtype=torch.float32).view(1, 1, -1, 1))
        self.register_buffer('h1_v', torch.tensor(h1, dtype=torch.float32).view(1, 1, -1, 1))

    def forward(self, x):
        B, C, H, W = x.shape

        if self.mode == 'zero':
            pad_w = self.h0_h.size(-1) - 1
            x_padded = F.pad(x, (pad_w, 0), mode='constant', value=0)
            _, _, H_padded, W_padded = x_padded.shape
        else:
            x_padded = x
            H_padded, W_padded = H, W

        x_reshaped = x_padded.view(B * C, 1, H_padded, W_padded)

        x_h0 = F.conv2d(x_reshaped, self.h0_h, stride=(1, 2))
        x_h1 = F.conv2d(x_reshaped, self.h1_h, stride=(1, 2))

        _, _, H_out, W_out = x_h0.shape
        x_h0 = x_h0.view(B, C, H_out, W_out)
        x_h1 = x_h1.view(B, C, H_out, W_out)

        if self.mode == 'zero':
            pad_h = self.h0_v.size(-2) - 1
            x_h0_padded = F.pad(x_h0, (0, 0, pad_h, 0), mode='constant', value=0)
            x_h1_padded = F.pad(x_h1, (0, 0, pad_h, 0), mode='constant', value=0)
            _, _, H_vert_padded, W_vert_padded = x_h0_padded.shape
        else:
            x_h0_padded = x_h0
            x_h1_padded = x_h1
            H_vert_padded, W_vert_padded = H_out, W_out

        x_h0_reshaped = x_h0_padded.view(B * C, 1, H_vert_padded, W_vert_padded)
        x_h1_reshaped = x_h1_padded.view(B * C, 1, H_vert_padded, W_vert_padded)

        LL = F.conv2d(x_h0_reshaped, self.h0_v, stride=(2, 1))
        LH = F.conv2d(x_h0_reshaped, self.h1_v, stride=(2, 1))
        HL = F.conv2d(x_h1_reshaped, self.h0_v, stride=(2, 1))
        HH = F.conv2d(x_h1_reshaped, self.h1_v, stride=(2, 1))

        _, _, H_final, W_final = LL.shape
        LL = LL.view(B, C, H_final, W_final)
        LH = LH.view(B, C, H_final, W_final)
        HL = HL.view(B, C, H_final, W_final)
        HH = HH.view(B, C, H_final, W_final)

        return LL, (LH, HL, HH)

class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        self.wt = DWT(mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        LL, (LH, HL, HH) = self.wt(x)
        x = torch.cat([LL, HL, LH, HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

# class HWD(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(HWD, self).__init__()
#         self.wt = DWTForward(J=1, mode='zero', wave='haar')
#         self.conv_bn_relu = nn.Sequential(
#             nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         yL, yH = self.wt(x)
#         y_HL = yH[0][:, :, 0, ::]
#         y_LH = yH[0][:, :, 1, ::]
#         y_HH = yH[0][:, :, 2, ::]
#         x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
#         x = self.conv_bn_relu(x)
#         return x

if __name__ == "__main__":
    # 如果GPU可用，将模块移动到 GPU
    device = torch.device("cuda"if torch.cuda.is_available() else"cpu")
    # 输入张量 (batch_size, height, width,channels)
    x = torch.randn(1, 32, 40, 40).to(device)
    # 初始化 HWD 模块
    dim = 32
    block = HWD(dim, dim)
    print(block)
    block = block.to(device)
    # 前向传播
    output = block(x)
    print("输入:", x.shape)
    print("输出:", output.shape)