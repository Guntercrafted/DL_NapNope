# app/ui.py
from __future__ import annotations
import sys, time, os, cv2, threading, random, glob
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QShortcut, QKeySequence

try:
    from playsound import playsound
    HAS_PLAYSOUND = True
except Exception:
    HAS_PLAYSOUND = False

from .pipeline import Pipeline         # ✅ ใช้ mediapipe pipeline
from .state_machine import StateMachine
from .logger import EventLogger
from .config import (
    # layout / ui sizes
    START_W, START_H,
    MARGIN, HSPACE, VSPACE,
    BOTTOM_H, LIVE_H,
    GAG_W, GAG_H, GAG_IMAGE_PATH,
    # camera / runtime (Pipeline ใช้เองอยู่แล้ว)
    CAM_INDEX, FLIP,
)

# ---------------------------
# Helpers
# ---------------------------

def cv_bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """Convert OpenCV BGR np.ndarray -> QImage (RGB)."""
    if bgr is None or bgr.size == 0:
        return QImage()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

def make_btn(text: str) -> QtWidgets.QPushButton:
    b = QtWidgets.QPushButton(text)
    b.setFixedHeight(44)
    b.setCursor(Qt.PointingHandCursor)
    b.setStyleSheet("""
        QPushButton { background:#2F2F3A; color:white; border:0; padding:10px 16px; border-radius:8px; }
        QPushButton:hover { background:#3A3A48; }
        QPushButton:pressed { background:#292935; }
    """)
    return b

def make_stat_label(txt: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(txt)
    lab.setStyleSheet("color:#DCDCDC;")
    return lab


# ---------------------------
# Main UI
# ---------------------------

class NapNopeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # ---- window ----
        self.setWindowTitle("Nap?Nope! Drowsiness Detection (Mediapipe Version)")
        self.resize(START_W, START_H)
        self.setMinimumSize(1024, 600)
        self.setStatusBar(QtWidgets.QStatusBar(self))

        # ---- core components ----
        self.pipe = Pipeline()            # Pipeline แบบ mediapipe
        self.fsm  = StateMachine()        # (ใช้งานได้หาก pipeline ส่ง triggered จาก FSM)
        self.log  = EventLogger()

        # ---- central layout ----
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QGridLayout(central)
        root.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        root.setHorizontalSpacing(HSPACE)
        root.setVerticalSpacing(VSPACE)

        # ===== Left: LIVE (fixed height 700) =====
        live_container = QtWidgets.QWidget()
        live_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        live_container.setFixedHeight(LIVE_H)
        live_layout = QtWidgets.QVBoxLayout(live_container)
        live_layout.setContentsMargins(0, 0, 0, 0)

        self.live_lbl = QtWidgets.QLabel()
        self.live_lbl.setAlignment(Qt.AlignCenter)
        self.live_lbl.setStyleSheet("background:#000; border-radius:10px;")
        self.live_lbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # แสดงแบบคงอัตราส่วน ไม่ครอป
        self.live_lbl.setScaledContents(False)
        live_layout.addWidget(self.live_lbl)

        # ===== Right: GAG (fixed 300x700) =====
        gag_container = QtWidgets.QWidget()
        gag_container.setFixedSize(GAG_W, GAG_H)
        gag_layout = QtWidgets.QVBoxLayout(gag_container)
        gag_layout.setContentsMargins(0, 0, 0, 0)

        self.gag_lbl = QtWidgets.QLabel()
        self.gag_lbl.setFixedSize(GAG_W, GAG_H)
        self.gag_lbl.setAlignment(Qt.AlignCenter)
        self.gag_lbl.setStyleSheet("background:#1E1E28; color:#B8B8C8; border-radius:10px;")
        gag_layout.addWidget(self.gag_lbl)

        # เตรียมรายการไฟล์ GAG
        self.gag_files = self._collect_gags()
        self._load_gag_initial()

        # ===== Bottom controls =====
        controls = QtWidgets.QWidget()
        controls.setFixedHeight(BOTTOM_H)
        ctrl = QtWidgets.QHBoxLayout(controls)
        ctrl.setContentsMargins(0, 0, 0, 0)
        ctrl.setSpacing(12)

        self.btn_start = make_btn("► Start")
        self.btn_stop  = make_btn("▮ Stop")
        self.btn_save  = make_btn("📸 Save")
        self.btn_exit  = make_btn("✖ Exit")

        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        ctrl.addWidget(self.btn_save)
        ctrl.addWidget(self.btn_exit)

        # ---- status labels ----
        ctrl.addSpacing(18)
        self.lbl_eye  = make_stat_label("Eye: –")
        self.lbl_mou  = make_stat_label("Mouth: –")
        self.lbl_head = make_stat_label("Head: –")
        self.lbl_ear  = make_stat_label("EAR: 0.000")
        self.lbl_mar  = make_stat_label("MAR: 0.000")
        self.lbl_hr   = make_stat_label("HeadRatio: 0.00")

        for w in (self.lbl_eye, self.lbl_mou, self.lbl_head, self.lbl_ear, self.lbl_mar, self.lbl_hr):
            ctrl.addWidget(w)

        ctrl.addStretch(1)

        # ===== Spacer (ขวา) =====
        spacer_bottom = QtWidgets.QWidget()
        spacer_bottom.setFixedHeight(BOTTOM_H)
        spacer_bottom.setFixedWidth(GAG_W)

        # ---- layout grid ----
        root.addWidget(live_container, 0, 0, 1, 1)
        root.addWidget(gag_container,  0, 1, 1, 1)
        root.addWidget(controls,       1, 0, 1, 1)
        root.addWidget(spacer_bottom,  1, 1, 1, 1)
        root.setRowStretch(0, 0)
        root.setRowStretch(1, 0)
        root.setColumnStretch(0, 1)
        root.setColumnStretch(1, 0)

        # ---- connections ----
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_exit.clicked.connect(self.close_app)
        self.btn_save.clicked.connect(self.save_snapshot)

        if hasattr(self.pipe, "new_frame"):
            self.pipe.new_frame.connect(self.on_new_frame)
        elif hasattr(self.pipe, "frame_ready"):
            self.pipe.frame_ready.connect(self.on_new_frame)

        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.close_app)

        self._last_status_ts = 0.0
        self.statusBar().showMessage("Ready")

    # ---------------------------
    # GAG helpers
    # ---------------------------

    def _collect_gags(self) -> list[str]:
        """
        รวมไฟล์จากโฟลเดอร์ 'gag' ถ้ามี (รองรับ .png .jpg .jpeg).
        ถ้าไม่มี ให้ใช้ GAG_IMAGE_PATH จาก config เป็น default display.
        """
        files = []
        try:
            files = (
                glob.glob(os.path.join("gag", "*.png")) +
                glob.glob(os.path.join("gag", "*.jpg")) +
                glob.glob(os.path.join("gag", "*.jpeg"))
            )
            files.sort()
        except Exception:
            files = []
        return files

    def _set_gag_pixmap(self, path: str | None):
        if path and os.path.exists(path):
            pix = QtGui.QPixmap(path).scaled(GAG_W, GAG_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.gag_lbl.setPixmap(pix)
        elif os.path.exists(GAG_IMAGE_PATH):
            pix = QtGui.QPixmap(GAG_IMAGE_PATH).scaled(GAG_W, GAG_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.gag_lbl.setPixmap(pix)
        else:
            self.gag_lbl.setText("Gag (300×700)\nnot found\n→ assets/gag.png")

    def _load_gag_initial(self):
        # ถ้ามีหลายไฟล์ใน gag/ ใช้อันแรกเป็นค่าเริ่มต้น ไม่งั้นใช้ GAG_IMAGE_PATH
        path = self.gag_files[0] if self.gag_files else GAG_IMAGE_PATH
        self._set_gag_pixmap(path)

    def _swap_gag_random(self):
        if not self.gag_files:
            # ไม่มีรูปในโฟลเดอร์ gag → ใช้ default
            self._set_gag_pixmap(GAG_IMAGE_PATH)
            return
        self._set_gag_pixmap(random.choice(self.gag_files))

    # ---------------------------
    # Buttons
    # ---------------------------

    def start_detection(self):
        try:
            self.pipe.start()
            self.statusBar().showMessage("Detection started…")
        except Exception as e:
            self.statusBar().showMessage(f"Start error: {e}")

    def stop_detection(self):
        try:
            self.pipe.stop()
            self.statusBar().showMessage("Detection stopped.")
        except Exception as e:
            self.statusBar().showMessage(f"Stop error: {e}")

    def save_snapshot(self):
        frame = getattr(self.pipe, "last_frame", None)
        if frame is None:
            self.statusBar().showMessage("No frame to save.")
            return
        img = cv_bgr_to_qimage(frame)
        pix = QPixmap.fromImage(img)
        os.makedirs("snapshots", exist_ok=True)
        fn = time.strftime("snapshots/%Y%m%d_%H%M%S.png")
        pix.save(fn)
        self.statusBar().showMessage(f"Saved {fn}", 3000)

    def close_app(self):
        try:
            self.pipe.stop()
        except Exception:
            pass
        self.close()

    # ---------------------------
    # Sound (async)
    # ---------------------------

    def _play_alert_sound(self, path: str):
        if not HAS_PLAYSOUND or not os.path.exists(path):
            return
        threading.Thread(target=lambda: playsound(path), daemon=True).start()

    # ---------------------------
    # Slot: receive frames
    # ---------------------------

    @QtCore.Slot(object, dict)
    def on_new_frame(self, frame, info):
        # --- แสดงภาพแบบคงอัตราส่วน (Letterbox) ---
        qimg = cv_bgr_to_qimage(frame)
        pix = QPixmap.fromImage(qimg).scaled(
            self.live_lbl.width(), self.live_lbl.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.live_lbl.setPixmap(pix)

        # --- อัปเดตค่าด้านล่าง ---
        eye_state   = str(info.get("eye_state", "–")).upper()
        mouth_state = str(info.get("mouth_state", "–")).upper()
        head_state  = str(info.get("head_state", "–")).upper()
        ear         = float(info.get("ear", 0.0))
        mar         = float(info.get("mar", 0.0))
        head_ratio  = float(info.get("head_ratio", 0.0))
        triggered   = info.get("triggered")

        self.lbl_eye.setText (f"Eye: {eye_state}")
        self.lbl_mou.setText (f"Mouth: {mouth_state}")
        self.lbl_head.setText(f"Head: {head_state}")
        self.lbl_ear.setText (f"EAR: {ear:.3f}")
        self.lbl_mar.setText (f"MAR: {mar:.3f}")
        self.lbl_hr.setText  (f"HeadRatio: {head_ratio:.2f}")

        # --- เมื่อมี Alert ---
        if triggered:
            # 1) บันทึก Event
            self.log.log(triggered, info)
            # 2) แสดงใน status bar (กันสแปมทุก 0.5s)
            now = time.time()
            if self._last_status_ts + 0.5 <= now:
                self.statusBar().showMessage(f"⚠ {triggered}", 2000)
                self._last_status_ts = now
            # 3) เล่นเสียง
            self._play_alert_sound(os.path.join("notification", "sound_notification.mp3"))
            # 4) เปลี่ยนรูป gag แบบสุ่ม
            self._swap_gag_random()


# ---------------------------
# Entrypoint
# ---------------------------

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = NapNopeApp()
    win.show()
    sys.exit(app.exec())
