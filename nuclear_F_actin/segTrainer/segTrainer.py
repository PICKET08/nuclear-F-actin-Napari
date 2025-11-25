import os
import time
from importlib.util import find_spec

def check_device(torch):
    train_on_gpu = torch.cuda.is_available()
    print('CUDA is available. Training on GPU' if train_on_gpu else 'CUDA is not available. Training on CPU')
    return torch.device("cuda:0" if train_on_gpu else "cpu"), train_on_gpu

def create_data_loaders(train_data_dir, train_label_dir, batch_size, num_workers, pin_memory, torch, load):

    train_data = load(train_data_dir, train_label_dir)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader

def train_epoch(model, train_loader, optimizer, device, n_iter, calc_loss):

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

def train_model(model, train_loader, optimizer, scheduler, device, epochs, base_dir, segmodel, calc_loss, torch, log_fn=None):
    n_iter = 1
    for epoch in range(epochs):
        train_loss, n_iter = train_epoch(model, train_loader, optimizer, device, n_iter, calc_loss)
        scheduler.step()
        log_fn(f"Epoch {epoch}, Loss: {train_loss:.4f}", "info")
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
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
    if find_spec("torch") is None:
        log_fn("Torch is not available.", "warn")
        log_fn("Please wait while torch is installed...", "info")
        os.system("ltt install torch==2.8.0+cpu torchvision==0.23.0+cpu")

    import torch
    from .models.Unets import NestedUNet, U_Net
    from .models.fcn import FCN8s
    from .models.espnet import ESPNetV2
    from .models.icnet import ICNet
    from .models.deeplabv3 import DeepLabV3
    from .models.bisenet import BiSeNet

    from .tools.Data_Loader import Images_Dataset_folder, PadToSize, remove_padding
    from .tools.losses import calc_loss

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

    device, train_on_gpu = check_device(torch)

    if model_maps[segmodel] in {FCN8s, BiSeNet, ESPNetV2, ICNet, DeepLabV3}:
        model = model_maps[segmodel](1)
    else:
        model = model_maps[segmodel](3, 1)
    model.to(device)

    train_loader = create_data_loaders(
        train_data_dir, train_label_dir, batch_size, num_workers, train_on_gpu, torch, Images_Dataset_folder
    )

    initial_lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e10), eta_min=1e-5)

    model = train_model(
        model, train_loader, optimizer, scheduler, device, epochs, base_dir, segmodel, calc_loss, torch, log_fn)

    if train_on_gpu:
        torch.cuda.empty_cache()
