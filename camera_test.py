import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open /dev/video0")
else:
    print("✅ Camera opened successfully")

cap.release()
