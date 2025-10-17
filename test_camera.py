import cv2

CAM_INDEX = 0  # เริ่มจากกล้องหลัก notebook

print("🎥 กำลังเปิดกล้อง... กด 'q' เพื่อออก")

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print(f"❌ เปิดกล้องไม่สำเร็จที่ index {CAM_INDEX}")
else:
    print("✅ เปิดกล้องสำเร็จ! แสดงภาพเรียลไทม์...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ ไม่สามารถอ่านภาพจากกล้องได้")
        break

    cv2.imshow("Camera Test", frame)

    # ออกจาก loop ด้วย q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🚪 กล้องถูกปิดเรียบร้อย")