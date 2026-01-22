import cv2
import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
from pipeline.anpr_pipeline import start_pipeline, dispatch_frame

# ===============================
# START PIPELINE
# ===============================
start_pipeline()

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("✅ ANPR system running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Non-blocking dispatch
    dispatch_frame(frame)

    cv2.imshow("Live Camera", frame)

    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
