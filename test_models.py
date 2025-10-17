import tensorflow as tf
import numpy as np
import os

# ===== โมเดลที่จะทดสอบ =====
MODELS = [
    "models/eyes_effv2_new.keras",
    "models/mouth_effv2_new.keras",
    "models/head_effv2_new.keras",
]

print("🧠 เริ่มทดสอบโมเดล...\n")

for path in MODELS:
    if not os.path.exists(path):
        print(f"❌ ไม่พบไฟล์: {path}")
        continue

    print(f"🔹 Loading model: {path}")
    model = tf.keras.models.load_model(path, compile=False)

    # ดึง input shape
    shape = model.input_shape
    h, w, c = shape[1:]
    print(f"   ↳ Input shape: ({h}, {w}, {c})")

    # สร้างภาพจำลอง (random noise) ขนาดที่โมเดลต้องการ
    dummy = np.random.rand(1, h, w, c).astype("float32")

    # รัน predict
    pred = model.predict(dummy, verbose=0)
    print(f"   ↳ Output shape: {pred.shape}")
    print(f"   ↳ Output preview: {pred.flatten()[:5]}\n")

print("✅ ทดสอบเสร็จสิ้น")
