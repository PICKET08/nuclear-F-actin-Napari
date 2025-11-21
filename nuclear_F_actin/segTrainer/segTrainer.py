import os
import time
import torch
import torch.utils.data
import torch.optim as optim
from nuclear_F_actin.segTrainer.models.Unets import NestedUNet, U_Net
from nuclear_F_actin.segTrainer.models.fcn import FCN8s
from nuclear_F_actin.segTrainer.models.espnet import ESPNetV2
from nuclear_F_actin.segTrainer.models.icnet import ICNet
from nuclear_F_actin.segTrainer.models.deeplabv3 import DeepLabV3
from nuclear_F_actin.segTrainer.models.bisenet import BiSeNet

from nuclear_F_actin.segTrainer.tools.Data_Loader import Images_Dataset_folder, PadToSize, remove_padding
from nuclear_F_actin.segTrainer.tools.losses import calc_loss

def check_device():
    train_on_gpu = torch.cuda.is_available()
    print('CUDA is available. Training on GPU' if train_on_gpu else 'CUDA is not available. Training on CPU')
    return torch.device("cuda:0" if train_on_gpu else "cpu"), train_on_gpu

def initialize_model(model_class, in_channel=3, out_channel=1, device='cpu'):
    if model_class in {FCN8s, BiSeNet, ESPNetV2, ICNet, DeepLabV3}:
        model = model_class(out_channel)
    else:
        model = model_class(in_channel, out_channel)
    model.to(device)
    print(f"Model {model_class.__name__} initialized and moved to {device}")
    return model

def create_data_loaders(train_data_dir, train_label_dir, batch_size, num_workers, pin_memory):

    train_data = Images_Dataset_folder(train_data_dir, train_label_dir)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader

def train_epoch(model, train_loader, optimizer, device, n_iter, epoch_idx):

    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = calc_loss(y_pred, y)
        train_loss += loss.item() * x.size(0)
        loss.backward()
        optimizer.step()

    return train_loss / len(train_loader.sampler), n_iter + 1

def train_model(model, train_loader, optimizer, scheduler, device, epochs, base_dir, segmodel, log_fn=None):
    n_iter = 1
    for epoch in range(epochs):
        train_loss, n_iter = train_epoch(model, train_loader, optimizer, device, n_iter, epoch)
        scheduler.step()
        log_fn(f"Epoch {epoch}, Loss: {train_loss:.4f}")
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # 示例输入，可调整为训练图像的标准尺寸
    seg_path = os.path.join(base_dir, "models", "seg")
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)
    now = time.localtime()
    time_str = time.strftime("%m%d%H%M", now)
    onnx_path = os.path.join(seg_path, segmodel + time_str + ".onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        })
    log_fn("Train Finished!")

    return model

def segtrain(epochs, segmodel, base_dir, train_data_dir, train_label_dir, log_fn=None):

    model_maps = {
        "U_Net": U_Net,
        "NestedUNet": NestedUNet,
        "DeepLabV3": DeepLabV3,
        "ICNet": ICNet,
        "ESPNetV2": ESPNetV2,
        "BiSeNet": BiSeNet,
        "FCN8s": FCN8s,
    }
    batch_size = 4
    num_workers = 0

    device, train_on_gpu = check_device()

    model = initialize_model(model_maps[segmodel], in_channel=3, out_channel=1, device=device)

    train_loader = create_data_loaders(
        train_data_dir, train_label_dir, batch_size, num_workers, train_on_gpu
    )

    initial_lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e10), eta_min=1e-5)

    model = train_model(
        model, train_loader, optimizer, scheduler, device, epochs, base_dir, segmodel, log_fn)

    if train_on_gpu:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    base_dir = "C:/Users/Administrator/Desktop/ImageJ"
    train_data_dir = "F:/Cell-Data/Seg/train-UNet/FN600/val/images"
    train_label_dir = "F:/Cell-Data/Seg/train-UNet/FN600/val/masks"
    segtrain(1, "FCN8s", base_dir, train_data_dir, train_label_dir)