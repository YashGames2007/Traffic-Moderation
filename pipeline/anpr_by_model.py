import cv2
from ultralytics import YOLO
import easyocr
import re
import os
import uuid

model = YOLO("model/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

DEFAULT_OUTPUT_DIR = "output"

def detect_by_model(img, output_path=None):
    """
    Runs YOLO + OCR on the image.

    Args:
        img (np.ndarray)
        output_path (str | None):
            - None → no file saved
            - directory path → auto filename
            - full file path → saved exactly there

    Returns:
        list[str]: detected plate strings
    """

    plates = []
    detected = False

    results = model(img, conf=0.10, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = img[y1:y2, x1:x2]
            text = reader.readtext(crop, detail=0)
            raw_text = " ".join(text)

            plate = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
            if plate:
                plates.append(plate)

            # draw annotations only if saving
            if output_path:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    plate,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # =========================
    # SAVE OUTPUT IMAGE
    # =========================
    if detected and output_path:
        # Case 1: output_path is a directory
        if os.path.isdir(output_path) or output_path.endswith(os.sep):
            os.makedirs(output_path, exist_ok=True)
            filename = f"anpr_{uuid.uuid4().hex[:8]}.jpg"
            save_path = os.path.join(output_path, filename)

        # Case 2: output_path is a file path
        else:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            save_path = output_path

        cv2.imwrite(save_path, img)

    return plates
