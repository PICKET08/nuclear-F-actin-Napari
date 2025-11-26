# Napari_Factin/weights.py
from qtpy.QtWidgets import (
    QWidget, QLabel, QComboBox, QRadioButton, QButtonGroup,
    QDoubleSpinBox, QTextEdit,  QPushButton,
    QLineEdit, QFileDialog, QApplication, QFrame, QSpinBox
)
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QFont

import os
import csv
import json
import time
import napari
import platformdirs
import numpy as np
from pathlib import Path
from napari.layers import Image

from .detect_inference import DetectONNX
from .segment_inference import SegmentONNX

from nuclear_f_actin.segTrainer.segTrainer import segtrain
from nuclear_f_actin.detTrainer.detTrainer import detTrain

class WeightsWindow(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self.viewer = viewer

        self.mode = 1
        self.mutiple = 0
        self.train_mode = 0
        self.epoch = 30
        self.input_size = 640

        self.det_model_dir = None
        self.seg_model_dir = None
        self.train_image_dir = None
        self.train_label_dir = None
        self.train_model = None
        self.csv_path = None
        self.detector = None
        self.segmentor = None

        self.config_dir = platformdirs.user_config_dir("Napari_Factin", "HIO")
        self.config_path = Path(self.config_dir) / "config.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self.setWindowTitle("Weights Manager")
        self.setFixedSize(450, 710)

        self.browse_button = QPushButton("Base Dir", self)
        self.browse_button.setGeometry(35, 30, 105, 25)
        self.browse_button.clicked.connect(self.select_base_dir)

        self.file_path_edit = QLineEdit(self)
        self.file_path_edit.setGeometry(150, 30, 230, 25)
        self.file_path_edit.setPlaceholderText("Select a Model Dir...")
        self.file_path_edit.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # QLabel 1 - Det Model
        self.detModelLabel = QLabel("Det Model", self)
        self.detModelLabel.setGeometry(35, 60, 105, 25)
        self.detModelLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # QLabel 2 - Seg Model
        self.segModelLabel = QLabel("Seg Model", self)
        self.segModelLabel.setGeometry(35, 90, 105, 25)
        self.segModelLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # QLabel 3 - Mode
        self.modeLabel = QLabel("Input Mode", self)
        self.modeLabel.setGeometry(35, 180, 105, 25)
        self.modeLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.modeLabel_2 = QLabel("Process Mode", self)
        self.modeLabel_2.setGeometry(35, 210, 105, 25)
        self.modeLabel_2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # QLabel 4 - Postprocessing
        self.processLabel = QLabel("Postprocessing", self)
        self.processLabel.setGeometry(35, 150, 105, 25)
        self.processLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # QLabel 5 - Magnification
        self.magLabel = QLabel("Magnification", self)
        self.magLabel.setGeometry(35, 120, 105, 25)
        self.magLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # ComboBox 1 - Select Det Model
        self.detComboBox = QComboBox(self)
        self.detComboBox.setGeometry(150, 60, 230, 25)
        self.detComboBox.addItem("Select Det Model")
        self.detComboBox.currentTextChanged.connect(self.select_det_model)

        # ComboBox 2 - Select Seg Model
        self.segComboBox = QComboBox(self)
        self.segComboBox.setGeometry(150, 90, 230, 25)
        self.segComboBox.addItem("Select Seg Model")
        self.segComboBox.currentTextChanged.connect(self.select_seg_model)

        # Radio Buttons - Mode Selection
        self.radioDS = QRadioButton("Det and Seg", self)
        self.radioDS.setGeometry(150, 210, 115, 25)
        self.radioDS.setChecked(True)
        self.radioDS.toggled.connect(self.select_mode)

        self.radioOD = QRadioButton("Only Det", self)
        self.radioOD.setGeometry(265, 210, 115, 25)
        self.radioOD.toggled.connect(self.select_mode)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.radioDS)
        self.mode_group.addButton(self.radioOD)

        # Radio Buttons - Process Mode Selection
        self.radioS = QRadioButton("Single", self)
        self.radioS.setGeometry(150, 180, 115, 25)
        self.radioS.setChecked(True)
        self.radioS.toggled.connect(self.select_mode_2)

        self.radioM = QRadioButton("Multiple", self)
        self.radioM.setGeometry(265, 180, 115, 25)
        self.radioM.toggled.connect(self.select_mode_2)

        self.mode_group_2 = QButtonGroup(self)
        self.mode_group_2.addButton(self.radioS)
        self.mode_group_2.addButton(self.radioM)

        # SpinBox - Postprocessing
        self.spin_box = QDoubleSpinBox(self)
        self.spin_box.setGeometry(150, 150, 230, 25)
        self.spin_box.setRange(0.0, 1.0)
        self.spin_box.setSingleStep(0.1)
        self.spin_box.setValue(0.4)
        self.spin_box.setAlignment(Qt.AlignCenter)
        self.spin_box.valueChanged.connect(self.select_threshold)

        # SpinBox - Magnification
        self.spin_box_2 = QDoubleSpinBox(self)
        self.spin_box_2.setGeometry(150, 120, 230, 25)
        self.spin_box_2.setRange(0.002, 200)
        self.spin_box_2.setSingleStep(1)
        self.spin_box_2.setValue(1)
        self.spin_box_2.setAlignment(Qt.AlignCenter)
        self.spin_box_2.valueChanged.connect(self.select_magnification)

        # QTextEdit - Output box (with scrollbar automatically)
        self.output_box = QTextEdit(self)
        self.output_box.setGeometry(35, 455, 370, 250)
        self.output_box.setPlaceholderText("Please select a model... \n")
        self.output_box.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.output_box.setReadOnly(True)

        # QPushButton - Run
        self.run_button = QPushButton("Infer", self)
        self.run_button.setGeometry(35, 245, 370, 25)
        self.run_button.clicked.connect(self.run)

        self.train_button = QPushButton("Train", self)
        self.train_button.setGeometry(35, 420, 370, 25)
        self.train_button.clicked.connect(self.train)

        self.train_area = QFrame(self)
        self.train_area.setGeometry(35, 280, 370, 130)
        self.train_area.setFrameShape(QFrame.StyledPanel)
        self.train_area.setFrameShadow(QFrame.Sunken)
        self.train_area.setLineWidth(2)
        self.train_area.setStyleSheet("""
            QFrame#TrainAreaFrame {
                border: 1px solid #cccccc;  
                border-radius: 6px;
                background: transparent; 
            }
        """)
        self.train_area.setObjectName("TrainAreaFrame")

        self.train_label = QLabel("Training", self.train_area)
        self.train_label.setGeometry(5, 0, 100, 25)

        self.train_mode_label = QLabel("Mode", self.train_area)
        self.train_mode_label.setGeometry(5, 30, 65, 25)

        self.radioTS = QRadioButton("Seg", self.train_area)
        self.radioTS.setGeometry(68, 30, 55, 25)
        self.radioTS.toggled.connect(self.select_train_mode)

        self.radioTD = QRadioButton("Det", self.train_area)
        self.radioTD.setGeometry(140, 30, 55, 25)
        self.radioTD.toggled.connect(self.select_train_mode)

        self.mode_group_3 = QButtonGroup(self.train_area)
        self.mode_group_3.addButton(self.radioTS)
        self.mode_group_3.addButton(self.radioTD)

        self.train_model_label = QLabel("Model", self.train_area)
        self.train_model_label.setGeometry(5, 60, 65, 25)

        self.trainComboBox = QComboBox(self.train_area)
        self.trainComboBox.setGeometry(68, 60, 127, 25)
        self.trainComboBox.addItem("Select Model")
        self.trainComboBox.currentTextChanged.connect(self.select_train_model)

        self.train_data_label = QLabel("Dataset", self.train_area)
        self.train_data_label.setGeometry(5, 90, 65, 25)

        self.image_button = QPushButton("images", self.train_area)
        self.image_button.setGeometry(68, 90, 55, 25)
        self.image_button.clicked.connect(self.select_image_dir)
        self.label_button = QPushButton("labels", self.train_area)
        self.label_button.setGeometry(140, 90, 55, 25)
        self.label_button.clicked.connect(self.select_label_dir)

        self.train_epoch_label = QLabel("Epoch", self.train_area)
        self.train_epoch_label.setGeometry(210, 30, 50, 25)

        self.spin_epoch_box = QSpinBox(self.train_area)
        self.spin_epoch_box.setGeometry(265, 30, 100, 25)
        self.spin_epoch_box.setRange(1, 200)
        self.spin_epoch_box.setSingleStep(5)
        self.spin_epoch_box.setValue(30)
        self.spin_epoch_box.setAlignment(Qt.AlignCenter)
        self.spin_epoch_box.valueChanged.connect(self.select_epoch)

        self.train_size_label = QLabel("Size", self.train_area)
        self.train_size_label.setGeometry(210, 60, 50, 25)

        self.spin_size_box = QSpinBox(self.train_area)
        self.spin_size_box.setGeometry(265, 60, 100, 25)
        self.spin_size_box.setRange(1, 3000)
        self.spin_size_box.setSingleStep(10)
        self.spin_size_box.setValue(640)
        self.spin_size_box.setAlignment(Qt.AlignCenter)
        self.spin_size_box.valueChanged.connect(self.select_size)

        font = QFont("Consolas", 9)

        for widget in [self.detModelLabel,  self.segModelLabel, self.modeLabel,   self.processLabel,
                       self.magLabel,       self.modeLabel_2,   self.train_label, self.train_mode_label,
                       self.detComboBox,    self.segComboBox,   self.trainComboBox,
                       self.radioDS,        self.radioOD,       self.radioS,      self.radioM,
                       self.radioTS,        self.radioTD,
                       self.spin_box,       self.spin_box_2,    self.output_box,
                       self.spin_epoch_box, self.spin_size_box,
                       self.run_button,     self.browse_button, self.train_button,
                       self.image_button,   self.label_button,
                       self.file_path_edit, self.train_model_label,
                       self.train_data_label, self.train_size_label, self.train_epoch_label]:

            widget.setFont(font)

        self.load_model_dir_from_config()


    def load_model_dir_from_config(self):
        folder_path = None
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                    folder_path = data.get("model_root", None)
                    if folder_path and not os.path.exists(folder_path):
                        folder_path = None
            except Exception:
                folder_path = None
        if folder_path:
            self.config_dir = folder_path
            self.update_model_dirs(folder_path)

    def select_base_dir(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            try:
                with open(self.config_path, "w") as f:
                    json.dump({"model_root": folder_path}, f)
            except Exception:
                pass
            self.config_dir = folder_path
            self.update_model_dirs(folder_path)


    def select_image_dir(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if folder_path:
            self.train_image_dir = folder_path
            self.logging("Train images dir: " + folder_path, "info")

    def select_label_dir(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Labels Folder")
        if folder_path:
            self.train_label_dir = folder_path
            self.logging("Train labels dir: " + folder_path,"info")

    def update_model_dirs(self, folder_path):
        self.file_path_edit.setText(folder_path)
        self.create_csv()
        self.det_model_dir = os.path.join(folder_path, "models\Det")
        self.seg_model_dir = os.path.join(folder_path, "models\Seg")

        if os.path.isdir(self.det_model_dir):
            det_onnx_files = [f for f in os.listdir(self.det_model_dir) if f.endswith(".onnx")]
            self.detComboBox.clear()
            self.detComboBox.addItems(['Select Det Model'])
            self.detComboBox.addItems(det_onnx_files)
        else:
            self.logging(f"Warning: 'Det' folder not found in: {folder_path}models", "warn")

        if os.path.isdir(self.seg_model_dir):
            seg_onnx_files = [f for f in os.listdir(self.seg_model_dir) if f.endswith(".onnx")]
            self.segComboBox.clear()
            self.segComboBox.addItems(['Select Seg Model'])
            self.segComboBox.addItems(seg_onnx_files)
        else:
            self.logging(f"Warning: 'Seg' folder not found in: {folder_path}\models", "warn")

    def create_csv(self):
        outputs_dir = os.path.join(self.config_dir, "output")
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)

        now_str = time.strftime("%Y%m%d%H%M", time.localtime())
        csv_filename = f"{now_str}.csv"
        self.csv_path = os.path.join(outputs_dir, csv_filename)

        title = ["Image","Cells", "F-actin", "Normal", "Ratio",
                 "[0.008,0.05)", "[0.005,0.1)", "[0.1,0.2)",
                 "[0.2,0.3)", "[0.3,0.4)", "[0.4,0.5)", "[0.5,1]"]
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(title)

    def select_det_model(self, det_model):
        if det_model.lower().endswith('.onnx'):
            det_model_path = os.path.join(self.det_model_dir, det_model)
            self.detector = DetectONNX(det_model_path)
            self.logging("Detect Model loaded: " + det_model + "!\n")

    def select_seg_model(self, seg_model):
        if seg_model.lower().endswith('.onnx'):
            seg_model_path = os.path.join(self.seg_model_dir, seg_model)
            self.segmentor = SegmentONNX(seg_model_path)
            self.logging("Segment Model loaded: " + seg_model + "!\n")

    def select_train_model(self, train_model):
         self.train_model = train_model

    def select_mode(self):
        if self.radioDS.isChecked():
            self.mode = 1
        elif self.radioOD.isChecked():
            self.mode = 2

    def select_mode_2(self):
        if self.radioS.isChecked():
            self.mutiple = 0
        elif self.radioM.isChecked():
            self.mutiple = 1

    def select_train_mode(self):
        if self.radioTS.isChecked():
            self.train_mode = 0
            model_list = ["U_Net",
                          "NestedUNet",
                          "DeepLabV3",
                          "ICNet",
                          "ESPNetV2",
                          "BiSeNet",
                          "FCN8s"]
            self.trainComboBox.clear()
            self.trainComboBox.addItems(model_list)
        elif self.radioTD.isChecked():
            self.train_mode = 1
            model_list = ["CHC-YOLO", "YOLO"]
            self.trainComboBox.clear()
            self.trainComboBox.addItems(model_list)

    def select_threshold(self, value):
        if self.segmentor:
            self.segmentor.threshold = value

    def select_epoch(self, value):
        self.epoch = value

    def select_size(self, value):
        self.input_size = value

    def select_magnification(self, value):
        if self.segmentor:
            self.segmentor.magnification = value
        if self.detector:
            self.detector.magnification = value

    def logging(self, message: str, level: str = "info"):
        if level.lower() == "error":
            self.output_box.append(f'<span style="color:red;">{message}</span>')
        elif level.lower() == "warn":
            self.output_box.append(f'<span style="color:orange;">{message}</span>')
        else:
            self.output_box.append(message)

    def type_switch(self, image):
        if not np.issubdtype(image.dtype, np.uint8):
            image = image.astype(np.float32)
            image = np.clip(image, 0, 1) if image.max() <= 1.0 else np.clip(image, 0, 255)
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        elif image.ndim == 3:
            h, w, c = image.shape
            if c == 1:
                image = np.concatenate([image] * 3, axis=-1)
            elif c >= 4:
                image = image[..., :3]
            elif c < 3:
                image = np.concatenate([image] * (3 // c + 1), axis=-1)[..., :3]
        else:
            self.logging(f"Error: Unsupported image shape: {image.shape}", "error")
        return image

    def csv_write(self, data_list, csv_path):
        with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(data_list)

    def infer(self, image, name):
        if (image is not None):
            if self.mode == 1:
                if self.detector and self.segmentor is not None:
                    self.detector.image = image
                    self.segmentor.image = image
                    self.logging("Starting detection and segmentation...")
                    QApplication.processEvents()
                    image_segmented, csvList = self.segmentor.draw(*self.detector.crop())
                    csvList.insert(0, name)
                    self.logging("Segmentation completed!\n")
                    self.viewer.add_image(image_segmented, name=f"{name}_segmented")
                else:
                    self.logging("Error: No Model loaded!\n", "error")

            elif self.mode == 2:
                if self.detector is not None:
                    self.detector.image = image
                    self.logging("Starting object detection...")
                    image_detected, csvList = self.detector.draw()
                    csvList.insert(0, name)
                    self.logging("Detection completed!\n")
                    self.viewer.add_image(image_detected, name=f"{name}_detected")
                else:
                    self.logging("Error: No Model loaded!\n", "error")

            self.csv_write(csvList, self.csv_path)
        else:
            self.logging("Error: No image loaded!\n", "error")

    def run(self):
        self.run_button.setEnabled(False)
        self.run_button.setText("Processing...")
        try:
            image = None
            name = None
            if (self.mutiple == 0):
                for layer in reversed(self.viewer.layers):
                    if isinstance(layer, Image) and layer.visible:
                        name = layer.name
                        image = layer.data
                        image = self.type_switch(image)
                        break

                self.infer(image, name)

            if(self.mutiple == 1):
                for layer in reversed(self.viewer.layers):
                    if isinstance(layer, Image) and layer.visible:
                        name = layer.name
                        image = layer.data
                        image = self.type_switch(image)
                        self.infer(image, name)

        except Exception as e:
            self.logging(f"[Run] Error occurred: {e}", "error")
            import traceback
            traceback.print_exc()
        finally:
            self.run_button.setEnabled(True)
            self.run_button.setText("Infer")

    def train(self):
        self.train_button.setEnabled(False)
        self.train_button.setText("Training...")
        self.logging("Training start:")
        self.train_thread = TrainThread(self.train_mode, self.epoch, self.input_size, self.train_model, self.config_dir, self.train_image_dir,
                                        self.train_label_dir, self.logging)
        self.train_thread.finished.connect(self.train_finished)
        self.train_thread.start()

    def train_finished(self):
        self.train_button.setEnabled(True)
        self.train_button.setText("Train")
class TrainThread(QThread):
    finished = Signal()

    def __init__(self, train_mode, epoch, size, train_model, config_dir, train_image_dir, train_label_dir, log_fn=None):
        super().__init__()
        self.train_mode = train_mode
        self.epoch = epoch
        self.input_size = size
        self.train_model = train_model
        self.config_dir = config_dir
        self.train_image_dir = train_image_dir
        self.train_label_dir = train_label_dir
        self.log_fn = log_fn

    def run(self):
        if(self.train_mode == 0):
            try:
                segtrain(self.epoch, self.train_model, self.config_dir, self.train_image_dir, self.train_label_dir, self.log_fn)
            except Exception as e:
                self.log_fn(e, 'Error')
        elif(self.train_mode == 1):
            try:
                detTrain(self.epoch, self.train_model, self.input_size, self.config_dir, self.train_image_dir, self.log_fn)
            except Exception as e:
                self.log_fn(e, 'Error')
        self.finished.emit()




