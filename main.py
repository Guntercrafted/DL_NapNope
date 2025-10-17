# main.py
from PySide6 import QtWidgets
from app.ui import NapNopeApp
import sys

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)      
    win = NapNopeApp()
    win.show()
    sys.exit(app.exec())    