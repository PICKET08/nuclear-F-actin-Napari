from napari.utils.notifications import show_info
from qtpy.QtWidgets import QApplication

def show_hello_message():
    font = QApplication.font()
    show_info(font.family())
