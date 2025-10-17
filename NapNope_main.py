import cv2
import math
import numpy as np
import mediapipe as mp
from datetime import datetime

# ===== Config =====
EAR_THRESH   = 0.25   # เกณฑ์ตาปิด
FRAME_CHECK  = 20     # ต้องต่ำกว่าเกณฑ์ติดกันกี่เฟรมถึงเตือนหลับตา

MOUTH_THR    = 0.35   # เกณฑ์หาว (MAR > เกณฑ์)
YAWN_CHECK   = 12     # ต้องเกินเกณฑ์ติดกันกี่เฟรมถึงเตือนหาว

CAM_INDEX    = 0

LEFT_EYE_IDXS  = [33,160,158,133,153,144]
RIGHT_EYE_IDXS = [263,387,385,362,380,373]

# ใช้จุดมาตรฐานแก้ปัญหา MAR สูงผิดปกติ
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
MOUTH_UP_IN  = 13
MOUTH_DN_IN  = 14
MOUTH_UP_OUT = 81
MOUTH_DN_OUT = 178

mp_face = mp.solutions.face_mesh

def euclidean(p1, p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x*w, landmarks[i].y*h) for i in eye_indices]
    A = euclidean(pts[1], pts[5]); B = euclidean(pts[2], pts[4]); C = euclidean(pts[0], pts[3]) + 1e-6
    return (A+B)/(2.0*C), pts

def mouth_aspect_ratio(landmarks, w, h):
    # ใช้ 6 จุดชัดเจน
    L = (landmarks[MOUTH_LEFT].x*w,  landmarks[MOUTH_LEFT].y*h)
    R = (landmarks[MOUTH_RIGHT].x*w, landmarks[MOUTH_RIGHT].y*h)
    Uin = (landmarks[MOUTH_UP_IN].x*w,  landmarks[MOUTH_UP_IN].y*h)
    Din = (landmarks[MOUTH_DN_IN].x*w,  landmarks[MOUTH_DN_IN].y*h)
    Uo  = (landmarks[MOUTH_UP_OUT].x*w, landmarks[MOUTH_UP_OUT].y*h)
    Do  = (landmarks[MOUTH_DN_OUT].x*w, landmarks[MOUTH_DN_OUT].y*h)

    A = euclidean(Uin, Din)   # inner vertical
    B = euclidean(Uo,  Do)    # outer vertical
    C = euclidean(L,   R) + 1e-6

    mar = (A + B) / (2.0 * C)
    # สำหรับการวาด ให้คืน list จุดสำคัญ
    pts = [L, Uo, Uin, Din, Do, R]
    return mar, pts

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Cannot open camera"); return

    eye_flag = 0
    mouth_flag = 0
    last_event = "-"  # ใช้แสดงบนแผงล่าง

    try:
        import winsound
        def beep(): winsound.Beep(2500, 600)
    except Exception:
        def beep(): pass

    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                          min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            ear = 0.0; mar = 0.0
            eye_state = "OPEN"
            mouth_state = "NORMAL"

            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0].landmark

                # ----- Eyes -----
                lEAR, l_pts = eye_aspect_ratio(face, LEFT_EYE_IDXS, w, h)
                rEAR, r_pts = eye_aspect_ratio(face, RIGHT_EYE_IDXS, w, h)
                ear = (lEAR + rEAR)/2.0

                for pts in (l_pts, r_pts):
                    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, (0,255,0), 1)

                if ear < EAR_THRESH:
                    eye_flag += 1; eye_state = "CLOSED"
                else:
                    eye_flag = 0;  eye_state = "OPEN"

                # ----- Mouth -----
                mar, m_pts = mouth_aspect_ratio(face, w, h)
                # วาดปาก: เส้นกรอบ + กากบาท เว้นจุดให้ดูเข้าใจ
                L,Uo,Uin,Din,Do,R = [tuple(map(int,p)) for p in m_pts]
                cv2.polylines(frame, [np.array([L,Uo,R,Do], dtype=np.int32)], True, (0,200,255), 1)
                cv2.line(frame, Uin, Din, (0,200,255), 1)
                cv2.line(frame, L, R, (0,200,255), 1)

                if mar > MOUTH_THR:
                    mouth_flag += 1; mouth_state = "YAWN"
                else:
                    mouth_flag = 0;  mouth_state = "NORMAL"

                # ----- Alerts -----
                if eye_flag >= FRAME_CHECK:
                    last_event = "EYES CLOSED"
                    cv2.putText(frame, "ALERT: EYES CLOSED", (10, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    beep()

                if mouth_flag >= YAWN_CHECK:
                    last_event = "YAWN"
                    cv2.putText(frame, "ALERT: YAWN", (10, 125),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    beep()

                # ค่าไว้ดูบนตัวภาพ (จะย้ายออกจอก็ได้)
                cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(frame, f"MAR: {mar:.3f}", (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            else:
                eye_flag = mouth_flag = 0
                eye_state = "OPEN"; mouth_state = "NORMAL"
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # ===== Bottom panel (นอกวิดีโอ) 3 บรรทัด =====
            bar_h = 95
            canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
            canvas[:h, :, :] = frame
            cv2.rectangle(canvas, (0, h), (w, h + bar_h), (40, 40, 40), -1)

            # บรรทัด 1: สถานะของแต่ละอวัยวะ
            cv2.putText(canvas, f"Eye: {eye_state}   Mouth: {mouth_state}",
                        (20, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # บรรทัด 2: ค่า EAR/MAR ย่อ
            cv2.putText(canvas, f"EAR:{ear:.3f}   MAR:{mar:.3f}",
                        (20, h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 1)

            # บรรทัด 3: Event + Quit
            cv2.putText(canvas, f"Event: {last_event}",
                        (20, h + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
            cv2.putText(canvas, "[q] Quit", (w - 120, h + 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)

            cv2.imshow("Drowsiness Detection (Basic: Eyes + Mouth)", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()