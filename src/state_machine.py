import time
# app/state_machine.py
from .config import (
    CLOSED_EYE_MIN_FRAMES,
    YAWN_MIN_FRAMES,
    YAWN_BURST_FRAMES,
    HEAD_DOWN_MIN_FRAMES,
    ALERT_COOLDOWN_FRAMES,
    ALERT_COOLDOWN_SEC,     
)


class StateMachine:
    """ติดตามสถานะการง่วง / หาว / หลับตา / ก้มศีรษะ"""

    def __init__(self, logger=None):
        self.logger = logger
        self.reset()

    def reset(self):
        self.eye_counter = 0
        self.yawn_counter = 0
        self.head_counter = 0
        self.last_alert_time = 0
        self.current_state = "OK"

    def update(self, eye_state: str, mouth_state: str, head_state: str):
        """อัปเดตสถานะจากผลโมเดลในแต่ละเฟรม"""
        now = time.time()
        triggered = None

        # --- ตรวจการหลับตา ---
        if eye_state == "closed":
            self.eye_counter += 1
        else:
            self.eye_counter = 0

        if self.eye_counter >= CLOSED_EYE_MIN_FRAMES:
            triggered = "Drowsy (Eyes Closed)"
            self.eye_counter = 0

        # --- ตรวจการหาว ---
        if mouth_state == "yawn":
            self.yawn_counter += 1
        else:
            self.yawn_counter = 0

        if self.yawn_counter >= YAWN_MIN_FRAMES:
            triggered = "Yawning"
            self.yawn_counter = 0

        # --- ตรวจการก้มศีรษะ ---
        if head_state == "down":
            self.head_counter += 1
        else:
            self.head_counter = 0

        if self.head_counter >= HEAD_DOWN_MIN_FRAMES:
            triggered = "Head Down"
            self.head_counter = 0

        # --- ตรวจช่วง burst ของการหาว (กรณีพิเศษ) ---
        if triggered == "Yawning" and self.yawn_counter >= YAWN_BURST_FRAMES:
            triggered = "Repeated Yawning"

        # --- ตรวจการ Cooldown (ป้องกันเตือนรัวเกินไป) ---
        if triggered:
            if now - self.last_alert_time >= ALERT_COOLDOWN_SEC:
                self.last_alert_time = now
                self.current_state = triggered
                if self.logger:
                    self.logger.log("ALERT", triggered)
                print(f"⚠️ ALERT TRIGGERED: {triggered}")
            else:
                # ยังอยู่ในช่วงคูลดาวน์
                triggered = None

        # คืนค่าสถานะปัจจุบัน
        return self.current_state