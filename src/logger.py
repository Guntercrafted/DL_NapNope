import os, csv, datetime as dt
from .config import LOG_PATH, LOG_DIR

class EventLogger:
    """
    เขียนเหตุการณ์ลง logs/events.csv
    รูปแบบ: timestamp,event,eye_state,mouth_state,head_state,ear,mar,head_ratio
    """
    def __init__(self, path: str = LOG_PATH):
        self.path = path
        os.makedirs(LOG_DIR, exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp","event",
                            "eye_state","mouth_state","head_state",
                            "ear","mar","head_ratio"])

    def log(self, event: str, info: dict | None = None):
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info = info or {}
        row = [
            ts, event,
            info.get("eye_state",""),
            info.get("mouth_state",""),
            info.get("head_state",""),
            f'{info.get("ear",0):.3f}' if "ear" in info else "",
            f'{info.get("mar",0):.3f}' if "mar" in info else "",
            f'{info.get("head_ratio",0):.2f}' if "head_ratio" in info else "",
        ]
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
