import cv2, threading, time, random, os
import mediapipe as mp
import numpy as np
from PySide6.QtCore import QObject, Signal
from playsound import playsound  # ใช้เล่นเสียง
from .config import CAM_INDEX, FLIP


# ==============================
# Mediapipe-based Pipeline
# ==============================

def _distance(a, b):
    ax, ay = a
    bx, by = b
    return float(np.hypot(ax - bx, ay - by))


class Pipeline(QObject):
    new_frame = Signal(object, dict)   # ส่งภาพและข้อมูล
    drowsy_alert = Signal(str, str)    # (เหตุผล, path รูป GAG)

    def __init__(self, cam_index=CAM_INDEX, flip=FLIP):
        super().__init__()
        self.cam_index = cam_index
        self.flip = flip
        self.cap = None
        self.running = False
        self.last_frame = None

        # Mediapipe setup
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ----- HEAD reference (static horizontal line at nose level) -----
        self.ref_y = None
        self.ref_frames = 0
        self.ref_locked = False
        self.ref_lock_after = 30
        self.ref_alpha = 0.10

        # ----- ALERT SYSTEM -----
        self.eye_closed_start = None
        self.alert_cooldown = 5  # 5 วินาทีต่อการแจ้งเตือน
        self.last_alert_time = 0

        self.sound_path = os.path.join("notification", "sound_notification.mp3")
        self.gag_folder = os.path.join("gag")

    # ------------------------------
    # Start / Stop
    # ------------------------------
    def start(self):
        if self.running:
            return
        self.running = True
        self.ref_y = None
        self.ref_frames = 0
        self.ref_locked = False
        self.eye_closed_start = None
        self.last_alert_time = 0

        self.cap = cv2.VideoCapture(self.cam_index)
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.cap = None

    # ------------------------------
    # Frame Loop
    # ------------------------------
    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            if self.flip:
                frame = cv2.flip(frame, 1)

            self.last_frame = frame.copy()
            info = self._process_frame(frame)
            self.new_frame.emit(frame, info)
            time.sleep(0.03)

    # ------------------------------
    # Rotation Helper
    # ------------------------------
    @staticmethod
    def _rot(p, c, ang):
        x, y = p
        cx, cy = c
        ca, sa = np.cos(ang), np.sin(ang)
        xr = ca*(x - cx) - sa*(y - cy) + cx
        yr = sa*(x - cx) + ca*(y - cy) + cy
        return (xr, yr)

    # ------------------------------
    # Frame Processing
    # ------------------------------
    def _process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        ear = mar = head_ratio = 0.0
        eye_state = "unknown"
        mouth_state = "unknown"
        head_state = "unknown"
        triggered = None

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            pts = [(lm.x * w, lm.y * h) for lm in face.landmark]

            # ---- EAR (Eyes) ----
            LEFT = [33, 160, 158, 133, 153, 144]
            RIGHT = [362, 385, 387, 263, 373, 380]
            def calc_ear(idxs):
                A = _distance(pts[idxs[1]], pts[idxs[5]])
                B = _distance(pts[idxs[2]], pts[idxs[4]])
                C = _distance(pts[idxs[0]], pts[idxs[3]]) + 1e-6
                return (A + B) / (2.0 * C)
            ear = (calc_ear(LEFT) + calc_ear(RIGHT)) / 2.0
            eye_state = "closed" if ear < 0.22 else "open"

            # ---- MAR (Mouth) ----
            mar = _distance(pts[13], pts[14]) / (_distance(pts[78], pts[308]) + 1e-6)
            mouth_state = "yawn" if mar > 0.60 else "normal"

            # =========================================================
            # HEAD (deroll + static horizontal reference at nose level)
            # =========================================================
            L = pts[33]
            R = pts[263]
            nose = pts[1]
            cx, cy = ((L[0]+R[0])/2, (L[1]+R[1])/2)
            theta = np.arctan2(R[1]-L[1], R[0]-L[0])

            L_r = self._rot(L, (cx, cy), -theta)
            R_r = self._rot(R, (cx, cy), -theta)
            nose_r = self._rot(nose, (cx, cy), -theta)

            nose_y = float(nose_r[1])
            if not self.ref_locked:
                if self.ref_y is None:
                    self.ref_y = nose_y
                else:
                    self.ref_y = (1 - self.ref_alpha)*self.ref_y + self.ref_alpha*nose_y
                self.ref_frames += 1
                if self.ref_frames >= self.ref_lock_after:
                    self.ref_locked = True

            eye_dist = max(1.0, _distance(L_r, R_r))
            dy = (self.ref_y - nose_y) / eye_dist
            head_ratio = float(dy)

            UP_TH, DOWN_TH = +0.07, -0.07
            if head_ratio >= UP_TH:
                head_state = "up"
            elif head_ratio <= DOWN_TH:
                head_state = "down"
            else:
                head_state = "normal"

            # ---------- Alert Condition ----------
            now = time.time()
            if eye_state == "closed":
                if self.eye_closed_start is None:
                    self.eye_closed_start = now
                elif now - self.eye_closed_start >= 3.0:  # ปิดตาเกิน 3 วิ
                    if head_state == "down" or mouth_state == "yawn":
                        if now - self.last_alert_time >= self.alert_cooldown:
                            self.last_alert_time = now
                            triggered = "Drowsy Alert"
                            threading.Thread(target=self._alert_action, args=(triggered,), daemon=True).start()
            else:
                self.eye_closed_start = None

            # ---------- Draw Debug ----------
            x_left_r  = min(L_r[0], R_r[0]) - 15
            x_right_r = max(L_r[0], R_r[0]) + 15
            left_ref_r  = (x_left_r,  self.ref_y)
            right_ref_r = (x_right_r, self.ref_y)
            left_ref_orig  = self._rot(left_ref_r,  (cx, cy), +theta)
            right_ref_orig = self._rot(right_ref_r, (cx, cy), +theta)
            nose_orig      = self._rot(nose_r,      (cx, cy), +theta)
            cv2.line(frame, (int(left_ref_orig[0]), int(left_ref_orig[1])),
                     (int(right_ref_orig[0]), int(right_ref_orig[1])), (255, 200, 80), 2)
            cv2.circle(frame, (int(nose_orig[0]), int(nose_orig[1])), 5, (80, 255, 120), -1)
            cv2.putText(frame, f"HeadRatio: {head_ratio:+.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2, cv2.LINE_AA)

        return {
            "eye_state": eye_state,
            "mouth_state": mouth_state,
            "head_state": head_state,
            "ear": float(ear),
            "mar": float(mar),
            "head_ratio": float(head_ratio),
            "triggered": triggered
        }

    # ------------------------------
    # Alert actions
    # ------------------------------
    def _alert_action(self, reason="Drowsy Alert"):
        """เล่นเสียง + ส่งสัญญาณเปลี่ยนรูป GAG"""
        print(f"⚠ ALERT: {reason}")
        try:
            if os.path.exists(self.sound_path):
                threading.Thread(target=playsound, args=(self.sound_path,), daemon=True).start()
        except Exception as e:
            print("Sound error:", e)

        gag_path = self._get_random_gag()
        self.drowsy_alert.emit(reason, gag_path)

    def _get_random_gag(self):
        """สุ่มเลือกรูปจากโฟลเดอร์ gag"""
        if not os.path.exists(self.gag_folder):
            return ""
        imgs = [f for f in os.listdir(self.gag_folder) if f.endswith(".png")]
        if not imgs:
            return ""
        choice = random.choice(imgs)
        return os.path.join(self.gag_folder, choice)
