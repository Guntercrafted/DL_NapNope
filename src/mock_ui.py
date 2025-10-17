# app/mock_ui.py
# Realtime (preview) H = 700  |  GAG = 300 x 700

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QShortcut, QKeySequence
import os, time

# ---------- Config ----------
START_W, START_H = 1280, 800
MARGIN, HSPACE, VSPACE = 16, 16, 12
BOTTOM_H = 65

LIVE_H = 700              # ✅ จอกล้องสูงคงที่ 700
GAG_W, GAG_H = 300, 700   # ✅ GAG คงที่ 300x700
GAG_IMAGE_PATH = "assets/gag.png"

class MockApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nap?Nope! Drowsiness Detection (Mock — LIVE 700 / GAG 300x700)")
        self.resize(START_W, START_H)
        self.setMinimumSize(1024, 600)

        # ---------- Central Layout ----------
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QGridLayout(central)
        root.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        root.setHorizontalSpacing(HSPACE)
        root.setVerticalSpacing(VSPACE)

        # ---------- Realtime container (กำหนดความสูงคงที่) ----------
        live_container = QtWidgets.QWidget()
        live_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        live_container.setFixedHeight(LIVE_H)                  # ✅ สูง 700 ตลอด
        live_layout = QtWidgets.QVBoxLayout(live_container)
        live_layout.setContentsMargins(0, 0, 0, 0)

        self.live_lbl = QtWidgets.QLabel()
        self.live_lbl.setAlignment(Qt.AlignCenter)
        self.live_lbl.setStyleSheet("background:#000; border-radius:10px;")
        self.live_lbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        live_layout.addWidget(self.live_lbl)

        # ---------- GAG (ขวา) ----------
        gag_container = QtWidgets.QWidget()
        gag_container.setFixedSize(GAG_W, GAG_H)
        gag_layout = QtWidgets.QVBoxLayout(gag_container)
        gag_layout.setContentsMargins(0, 0, 0, 0)

        self.gag_lbl = QtWidgets.QLabel()
        self.gag_lbl.setFixedSize(GAG_W, GAG_H)
        self.gag_lbl.setAlignment(Qt.AlignCenter)
        self.gag_lbl.setStyleSheet("background:#1E1E28; color:#B8B8C8; border-radius:10px;")
        gag_layout.addWidget(self.gag_lbl)

        # ---------- แถบปุ่มล่าง ----------
        controls = QtWidgets.QWidget()
        controls.setFixedHeight(BOTTOM_H)
        ctrl = QtWidgets.QHBoxLayout(controls)
        ctrl.setContentsMargins(0, 0, 0, 0)
        ctrl.setSpacing(12)

        self.btn_start = QtWidgets.QPushButton("► Start")
        self.btn_stop  = QtWidgets.QPushButton("▮ Stop")
        self.btn_save  = QtWidgets.QPushButton("📸 Save")
        self.btn_exit  = QtWidgets.QPushButton("✖ Exit")
        for b in (self.btn_start, self.btn_stop, self.btn_save, self.btn_exit):
            b.setFixedHeight(44)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet("""
                QPushButton { background:#2F2F3A; color:white; border:0; padding:10px 16px; border-radius:8px; }
                QPushButton:hover { background:#3A3A48; }
                QPushButton:pressed { background:#292935; }
            """)

        self.info_lbl = QtWidgets.QLabel("Ready")
        self.info_lbl.setStyleSheet("color:#DCDCDC;")

        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        ctrl.addWidget(self.btn_save)
        ctrl.addWidget(self.btn_exit)
        ctrl.addSpacing(16)
        ctrl.addWidget(self.info_lbl, 1)

        spacer_bottom = QtWidgets.QWidget()
        spacer_bottom.setFixedHeight(BOTTOM_H)
        spacer_bottom.setFixedWidth(GAG_W)

        # ---------- วางลง Grid ----------
        root.addWidget(live_container, 0, 0, 1, 1)
        root.addWidget(gag_container,  0, 1, 1, 1)
        root.addWidget(controls,       1, 0, 1, 1)
        root.addWidget(spacer_bottom,  1, 1, 1, 1)
        root.setRowStretch(0, 0)   # แถวบนสูงคงที่ (เพราะ LIVE_H กำหนดอยู่แล้ว)
        root.setRowStretch(1, 0)
        root.setColumnStretch(0, 1)
        root.setColumnStretch(1, 0)

        # ---------- mock video ----------
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._draw_mock_frame)
        self.running = False
        self._t0 = time.time()

        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_exit.clicked.connect(self.close)
        QShortcut(QKeySequence(Qt.Key_F11), self, self.toggle_fullscreen)

        self._load_gag()

    def _load_gag(self):
        if os.path.exists(GAG_IMAGE_PATH):
            pix = QtGui.QPixmap(GAG_IMAGE_PATH).scaled(GAG_W, GAG_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.gag_lbl.setPixmap(pix)
        else:
            self.gag_lbl.setText("Gag Image (300×700)\nnot found\n→ assets/gag.png")

    def start(self):
        if self.running: return
        self.running = True
        self.timer.start()
        self.info_lbl.setText("Detection started… (mock)")

    def stop(self):
        if not self.running: return
        self.running = False
        self.timer.stop()
        self.info_lbl.setText("Detection stopped.")

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _draw_mock_frame(self):
        w, h = self.live_lbl.width(), self.live_lbl.height()
        img = QImage(w, h, QImage.Format_RGB32); img.fill(Qt.black)
        p = QPainter(img); p.setRenderHint(QPainter.Antialiasing, True)

        grad = QtGui.QLinearGradient(0, 0, w, h)
        grad.setColorAt(0.0, QColor(32, 32, 36)); grad.setColorAt(1.0, QColor(18, 18, 22))
        p.fillRect(0, 0, w, h, grad)

        t = time.time() - self._t0
        ear  = 0.25 + 0.10 * (0.5 + 0.5 * QtCore.qSin(2.3 * t))
        mar  = 0.45 + 0.25 * (0.5 + 0.5 * QtCore.qSin(1.7 * t + 1.2))
        head = 0.55 + 0.45 * (0.5 + 0.5 * QtCore.qSin(0.9 * t + 2.0))

        p.setFont(QFont("Segoe UI", 18)); p.setPen(QColor(255,255,255))
        p.drawText(24, 40, "Nap?Nope! Mock Preview (H=700)")

        p.fillRect(0, h-110, w, 110, QColor(0,0,0,160))
        p.setFont(QFont("Segoe UI", 16))
        p.drawText(24, h-70, f"Eye: {'OPEN' if ear>0.24 else 'CLOSED'} | "
                             f"Mouth: {'YAWN' if mar>0.6 else 'NORMAL'} | "
                             f"Head: {'DOWN' if head<0.5 else 'NORMAL'}")
        p.setFont(QFont("Segoe UI", 13))
        p.drawText(24, h-36, f"EAR: {ear:.3f} | MAR: {mar:.3f} | HeadScore: {head:.2f}")
        p.end()

        self.live_lbl.setPixmap(QtGui.QPixmap.fromImage(img))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = MockApp()
    win.show()
    sys.exit(app.exec())
