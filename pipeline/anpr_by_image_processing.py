import numpy as np
import cv2
import os
from pipeline.ocr_util import anpr_ocr


# Initialize EasyOCR reader ONCE (important for performance)
# reader = easyocr.Reader(['en'], gpu=False)


def detect_by_img_processing(
    img,
    output_dir="output",
    debug=False,
    debug_dir="debug_ip",
    frame_id=""
):
    """
    Classical image-processing based number plate detection
    using EasyOCR.
    """
    def save_debug(name, image):
        if debug:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, name), image)

    def ratioCheck(area, width, height):
        ratio = float(width) / float(height) if height != 0 else 0
        if ratio < 1:
            ratio = 1 / ratio
        if (area < 1063.62 or area > 73862.5) or (ratio < 3 or ratio > 6):
            return False
        return True

    def isMaxWhite(plate):
        return np.mean(plate) >= 115

    def ratio_and_rotation(rect):
        (_, _), (width, height), rect_angle = rect
        if width == 0 or height == 0:
            return False
        angle = -rect_angle if width > height else 90 + rect_angle
        if angle > 15:
            return False
        return ratioCheck(width * height, width, height)

    def clean2_plate(plate):
        gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
        save_debug("04_plate_thresh.jpg", thresh)

        contours, _ = cv2.findContours(
            thresh.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        if not ratioCheck(cv2.contourArea(cnt), w, h):
            return None

        return thresh[y:y+h, x:x+w]

    # -----------------------------
    # preprocessing
    # -----------------------------

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    save_debug("01_blur.jpg", blur)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    save_debug("02_gray.jpg", gray)

    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    save_debug("03_sobel.jpg", sobel)

    _, binary = cv2.threshold(
        sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    save_debug("04_binary.jpg", binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    save_debug("05_morph.jpg", morph)

    contours, _ = cv2.findContours(
        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    # -----------------------------
    # contour analysis
    # -----------------------------
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if not ratio_and_rotation(rect):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        plate_candidate = img[y:y+h, x:x+w]

        if not isMaxWhite(plate_candidate):
            continue

        clean_plate = clean2_plate(plate_candidate)
        if clean_plate is None:
            continue

        # -----------------------------
        # EasyOCR
        # -----------------------------
        # results = reader.readtext(
        #     clean_plate,
        #     detail=0,
        #     allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        # )

        # if not results:
        #     continue

        # text = results[0].strip()
        text = anpr_ocr(clean_plate)

        if not text:
            continue
        # -----------------------------
        # SAVE FINAL PLATE IMAGE
        # -----------------------------
        os.makedirs(output_dir, exist_ok=True)
        plate_path = os.path.join(
            output_dir, f"anpr_ip_{frame_id}_plate.jpg"
        )
        cv2.imwrite(plate_path, clean_plate)

        return text

    return None
