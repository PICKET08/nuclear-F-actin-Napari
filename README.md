# nuclear F-actin
Nuclear F-actin is an Napari plugin that implements a deep learning–based workflow for the quantitative analysis of nuclear F-actin in cells. The plugin follows a pipeline of cell detection and segmentation of specific intracellular components, designed to enable quantitative analysis of target features. Users can retrain the model using their own datasets.

# Installation
1. Set up the environment via command line.
      ```bash
      conda create -n nFA-napari python==3.9
      conda activate nFA-napari
      conda install pip
      ```
2. Install `nuclear F-actin`.
   
     i.via [PyPI]
     ```bash
     pip install nuclear_F_actin
     ```
     ii.via GitHub
     ```bash
     pip install git+
     ```
     iii.via ZIP file
     ```bash
     cd
     python -m pip install -e.
     ```
