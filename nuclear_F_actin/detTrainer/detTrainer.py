import os
import yaml
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_wavelets import DWTForward
from ultralytics import YOLO

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
    def __init__(self, out_ch):
        super(HWD, self).__init__()
        in_ch = int(out_ch / 2)
        self.wt = DWT(wave='haar', mode='zero')
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

        W = self.comp(X)
        W = self.enc(W)
        W = self.pix_shf(W)
        W = torch.softmax(W, dim=1)

        X = self.upsmp(X)
        X = self.unfold(X)
        X = X.view(b, c, -1, h_, w_)

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])
        return X

class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class CPCA_ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(CPCA_ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class CPCA(nn.Module):
    def __init__(self, channels, channelAttention_reduce=4):
        super().__init__()

        self.ca = CPCA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        inputs = self.ca(inputs)

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out

def register_custom_modules():
    try:
        from ultralytics.nn import modules
        import ultralytics.nn.tasks
        import sys

        custom_modules = {
            'CARAFE': CARAFE,
            'HWD': HWD,
            'CPCA': CPCA,
        }

        for module_name, module_class in custom_modules.items():
            setattr(modules, module_name, module_class)
            setattr(ultralytics.nn.tasks, module_name, module_class)

            ultralytics.nn.tasks.__dict__[module_name] = module_class
        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False

class EpochEndCallback:
    def __init__(self, log_fn=None):
        self.log_fn = log_fn

    def __call__(self, trainer):
        epoch = trainer.epoch
        loss = trainer.loss
        if self.log_fn:
            self.log_fn(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        else:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

class CustomYOLO:
    def __init__(self, model_path, log_fn=None):
        register_custom_modules()
        self.model = YOLO(model_path)
        self.epoch_end_callback = EpochEndCallback(log_fn)
        self.model.add_callback('on_train_epoch_end', self.epoch_end_callback)

    def train(self, onnx_export_path=None, **kwargs):
        results = self.model.train(**kwargs)

        if onnx_export_path:
            self.export_onnx(onnx_export_path)
        return results

    def export_onnx(self, export_path, imgsz=640, simplify=True):
        try:
            success = self.model.export(
                format='onnx',
                imgsz=imgsz,
                simplify=simplify,
                dynamic=True,
            )
            if export_path and export_path != success:
                import shutil
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                if os.path.exists(success):
                    shutil.move(success, export_path)
            return export_path if export_path else success
        except Exception as e:
            print(Exception)
            return None

def update_yaml(yaml_path, new_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    data['train'] = new_path
    data['val'] = new_path
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)

def detTrain(epochs, train_model, size, base_dir, train_data_dir, log_fn=None):

    data_path = os.path.join(os.path.dirname(__file__), 'celldata.yaml')
    model_path = os.path.join(os.path.dirname(__file__), train_model + '.yaml')

    update_yaml(data_path, train_data_dir)
    custom_yolo = CustomYOLO(model_path, log_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_args = {
        'data': data_path,
        'epochs': epochs,
        'imgsz': size,
        'batch': 2,
        'workers': 0,
        'device': device,
        'amp': False,
        'val': False,
        'save': False,
        'plots': False,
    }
    time_str = time.strftime("%m%d%H%M", time.localtime())
    onnx_path = os.path.join(base_dir, 'models/det/' + train_model + '_' + time_str + '.onnx')
    results = custom_yolo.train(onnx_path, **train_args)

if __name__ == "__main__":
    detTrain()