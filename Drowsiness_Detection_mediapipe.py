import cv2
import math
import numpy as np
import mediapipe as mp

# ---- Config ----
EAR_THRESH = 0.25       # เกณฑ์ตาปิด
FRAME_CHECK = 20        # ต้องต่ำกว่าเกณฑ์ติดกันกี่เฟรมถึงเตือน
CAM_INDEX = 0           # กล้องตัวที่ 0

# จุด landmark รอบตาจาก FaceMesh
LEFT_EYE_IDXS  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [263, 387, 385, 362, 380, 373]

mp_face = mp.solutions.face_mesh

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3]) + 1e-6
    return (A + B) / (2.0 * C), pts

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    flag = 0
    try:
        import winsound
        def beep(): winsound.Beep(2500, 700)
    except ImportError:
        def beep(): pass

    with mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # 1 = flip แนวนอน (ซ้าย↔ขวา)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0].landmark
                leftEAR, left_pts = eye_aspect_ratio(face, LEFT_EYE_IDXS, w, h)
                rightEAR, right_pts = eye_aspect_ratio(face, RIGHT_EYE_IDXS, w, h)
                ear = (leftEAR + rightEAR) / 2.0
                for pts in (left_pts, right_pts):
                    hull = np.array(pts, dtype=np.int32)
                    cv2.polylines(frame, [hull], isClosed=True, color=(0,255,0), thickness=1)
                cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                if ear < EAR_THRESH:
                    flag += 1
                    if flag >= FRAME_CHECK:
                        cv2.putText(frame, "**************** ALERT! ****************", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        beep()
                else:
                    flag = 0
            else:
                flag = 0
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.imshow("Drowsiness Detection (MediaPipe)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()