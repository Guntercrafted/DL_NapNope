# ============================================================
# CONFIGURATION — Nap?Nope! Drowsiness Detection (Mediapipe)
# ============================================================

# ---------------- CAMERA ----------------
CAM_INDEX = 0           # 0 = กล้องโน้ตบุ๊ก
FRAME_W  = 1280
FRAME_H  = 720
FLIP     = True         # กลับภาพแนวนอน (mirror)

# ---------------- UI LAYOUT ----------------
START_W, START_H   = 1400, 860   # ขนาดหน้าต่างเริ่มต้น
MARGIN, HSPACE, VSPACE = 16, 16, 12
BOTTOM_H = 65

LIVE_H = 700                     # ความสูงจอกล้องคงที่
GAG_W, GAG_H = 300, 700          # ขนาด GAG ตายตัว
GAG_IMAGE_PATH = "assets/gag.png"

# ---------------- MODEL INPUT SIZES ----------------
EYE_IMG_SIZE   = 224
MOUTH_IMG_SIZE = 160
HEAD_IMG_SIZE  = 224

# ---------------- THRESHOLDS (probabilities) ----------------
EYE_OPEN_THRESH   = 0.20   # >= ถือว่า OPEN
MOUTH_YAWN_THRESH = 0.45   # >= ถือว่า YAWN

# ---------------- LOGGING ----------------
SNAP_DIR = "snapshots"
LOG_DIR  = "logs"
LOG_FILE = "events.csv"
LOG_PATH = f"{LOG_DIR}/{LOG_FILE}"

# ---------------- DROWSINESS RULES ----------------
CLOSED_EYE_MIN_FRAMES = 15
YAWN_BURST_FRAMES     = 8
YAWN_MIN_FRAMES       = YAWN_BURST_FRAMES
HEAD_DOWN_MIN_FRAMES  = 12
ALERT_COOLDOWN_FRAMES = 30
ALERT_COOLDOWN_SEC    = 2.0

# ---------------- RUNTIME ----------------
TARGET_FPS = 30


# ============================================================
# MEDIAPIPE HEAD DETECTION PARAMETERS
# ============================================================

# ==== LANDMARK INDEX (Mediapipe FaceMesh) ====
NOSE_TIP_IDX        = 4     # ปลายจมูก
LEFT_EYE_OUTER_IDX  = 33    # หางตาซ้าย
RIGHT_EYE_OUTER_IDX = 263   # หางตาขวา

# ==== HEAD CALIBRATION & SMOOTHING ====
HEAD_CALIB_FRAMES   = 45    # เฉลี่ย ~1.5s (30fps)
HEAD_SMOOTH_ALPHA   = 0.7   # 0..1 (1=เนียนขึ้น,แต่ช้ากว่า)

# ==== HEAD PITCH THRESHOLDS ====
# offset_norm = (y_nose - y_mid) / face_height
# > 0 = จมูกต่ำกว่าเส้นกลาง (ก้ม), < 0 = จมูกสูงกว่าเส้นกลาง (เงย)
HEAD_PITCH_DELTA_DOWN = 0.08   # มากกว่าฐาน +0.08 = DOWN
HEAD_PITCH_DELTA_UP   = 0.08   # น้อยกว่าฐาน -0.08 = UP
# ===== MOUTH / YAWN (Mediapipe) =====
# inner-lip indices (Mediapipe FaceMesh):
TOP_INNER_LIP_IDX    = 13   # upper inner lip
BOTTOM_INNER_LIP_IDX = 14   # lower inner lip
LEFT_MOUTH_CORNER    = 61
RIGHT_MOUTH_CORNER   = 291

CHIN_IDX             = 152  # คาง ใช้คำนวณ jaw-open
NOSE_TIP_IDX         = 4    # ถ้ายังไม่มีบรรทัดนี้ด้านบน ให้คงไว้ (ใช้ร่วมกับหัว)

# smoothing
MAR_SMOOTH_ALPHA     = 0.6  # 0..1 (1 = เนียนขึ้น/ช้าลง)
JAW_SMOOTH_ALPHA     = 0.6

# thresholds (ปรับได้ภายหลัง)
MAR_YAWN_THRESH          = 0.62   # ยิ่งใหญ่ยิ่งอ้าปากมาก (สัดส่วนแนวตั้ง/แนวนอนปาก)
JAW_OPEN_RATIO_THRESH    = 0.085  # (nose–chin)/eye_distance
MOUTH_MIN_FRAMES_YAWN    = 6      # ต้องเกิน threshold ติดกันกี่เฟรมถึงจะว่า "Yawning"
# ===== HEAD ORIENTATION (ใช้ใน pipeline_mediapipe.py) =====
# dot product ระหว่างเวกเตอร์หน้าผาก→คาง กับแกนตั้ง
#   1.0 = ก้มสุด (หัวชี้ลง)
#   0.0 = เงยสุด (หัวขนานพื้น)
HEAD_DOWN_DOT = 0.80   # ≥ 0.8 ถือว่าก้ม
HEAD_UP_DOT   = 0.35   # ≤ 0.35 ถือว่าเงย

# ปรับได้ภายหลังถ้าตรวจผิด


# ============================================================
# (optional) Debug / Visualization
# ============================================================
SHOW_LANDMARKS = True  # แสดงจุด landmark บนจอ (True/False)