import os
import yaml
from pathlib import Path
from ultralytics import YOLO

def detTrain(epochs, train_model, size, base_dir, train_data_dir, log_fn):

    yaml_path = os.path.join(os.path.dirname(__file__), 'celldata.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    data['train'] = train_data_dir
    data['val'] = train_data_dir
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)

    model_file = f"{train_model}.yaml"
    model = YOLO(model_file)

    results = model.train(
        data='celldata.yaml',
        epochs=epochs,
        imgsz=size,
        exist_ok=True,
        verbose=True,
        val=False,
        save=False,
        plots=False,
    )

    onnx_path = base_dir + "/models/det"
    success = model.export(
        format='onnx',
        project = onnx_path,
        imgsz=size,
        simplify=True,
        dynamic=True,
        opset=11,
    )
    if onnx_path  and onnx_path  != success:
        import shutil
        os.makedirs(os.path.dirname(onnx_path ), exist_ok=True)
        if os.path.exists(success):
            shutil.move(success, onnx_path)

# 使用示例
if __name__ == "__main__":
    # 定义日志函数
    def my_log(msg):
        print(f"[LOG] {msg}")

    train_data_dir = r"C:\Users\Administrator\Desktop\ImageJ\results\2025102317\images"
    base_dir= r"C:\Users\Administrator\Desktop\ImageJ"

    # 调用训练函数
    detTrain(
        epochs=5,
        train_model="YOLO",
        size=640,
        base_dir=base_dir,
        train_data_dir=train_data_dir,
        log_fn=my_log
    )