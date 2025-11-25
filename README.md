# nuclear F-actin
Nuclear F-actin is an Napari plugin that implements a deep learning–based workflow for the quantitative analysis of nuclear F-actin in cells. The plugin follows a pipeline of cell detection and segmentation of specific intracellular components, designed to enable quantitative analysis of target features. Users can retrain the model using their own datasets.

## Installation
1. Set up the environment via command line.
      ```bash
      conda create -n nFA-napari python==3.9
      conda activate nFA-napari
      conda install pip
      ```
2. Install `nuclear F-actin`.
   
     i. via [PyPI](https://pypi.org/).
     ```bash
     pip install nuclear_F_actin
     ```
     ii. via GitHub.
     ```bash
     pip install git+https://github.com/PICKET08/nuclear-F-actin-Napari.git
     ```
     iii. via ZIP file.
     ```bash
     cd path/to/nuclear-F-actin-Napari
     python -m pip install -e.
     ```
## Usage
1. Download the [model package file](), unzip and place the model package into the work directory.
2. Launch the plugin via Command Line.
      ```bash
      conda activate nFA-napari
      napari
      ```
## Video tutorials


---

### Notes

1. If you want to use GPU, please make sure to install `torch` and `torchvision` versions that match your CUDA version.  
   Example for CUDA 12.6:
   ```bash
   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
   
