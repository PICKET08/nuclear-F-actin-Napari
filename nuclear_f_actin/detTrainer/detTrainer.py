import os
import sys
import yaml
import time
from importlib.util import find_spec

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

def update_yaml(yaml_path, new_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    data['train'] = new_path
    data['val'] = new_path
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)

def detTrain(epochs, train_model, size, base_dir, train_data_dir, log_fn=None):
    if find_spec("torch") is None:
        log_fn("Torch is not available.", "warn")
        log_fn("Please wait while torch is installed...", "info")
        os.system("ltt install torch==2.8.0+cpu torchvision==0.23.0+cpu")
        log_fn("Torch successfully installed!", "info")
        return

    import torch

    sys.path.append(os.path.dirname(__file__))
    from ultralytics import YOLO

    data_path = os.path.join(os.path.dirname(__file__), 'celldata.yaml')
    model_path = os.path.join(os.path.dirname(__file__), train_model + '.yaml')

    update_yaml(data_path, train_data_dir)

    model = YOLO(model_path)
    epoch_end_callback = EpochEndCallback(log_fn)
    model.add_callback('on_train_epoch_end', epoch_end_callback)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=size,
        device=device,
        val=False,
        save=False,
        plots=False,
        amp=False,
    )

    time_str = time.strftime("%m%d%H%M", time.localtime())
    onnx_path = os.path.join(base_dir, 'models/det/' + train_model + '_' + time_str + '.onnx')

    success = model.export(
        format='onnx',
        imgsz=size,
        simplify=True,
        dynamic=True,
    )

    if onnx_path and onnx_path != success:
        import shutil
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        if os.path.exists(success):
            shutil.move(success, onnx_path)

    log_fn("Train Finished!")

if __name__ == "__main__":
    detTrain()

