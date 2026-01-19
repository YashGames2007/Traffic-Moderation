import cv2
import time
import number_plate_detection

# from anpr_utils import number_plate_detection
# or paste the function above this code

DETECTION_INTERVAL = 5  # seconds

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Camera not accessible")
    exit()

last_detection_time = 0

print("ANPR started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Run ANPR every 5 seconds
    if current_time - last_detection_time >= DETECTION_INTERVAL:
        last_detection_time = current_time

        try:
            plate_text = number_plate_detection.number_plate_detection(frame)

            if plate_text:
                plate_text = plate_text.strip()
                if plate_text:
                    print(f"[ANPR] Detected Plate: {plate_text}")
            else:
                print("No Number Plate!")

        except Exception as e:
            print("[ANPR ERROR]", e)

    cv2.imshow("Live Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
