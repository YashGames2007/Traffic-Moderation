import re
import cv2
import numpy as np
import easyocr
from paddleocr import PaddleOCR

# -----------------------------
# OCR engines (init ONCE)
# -----------------------------
easy_reader = easyocr.Reader(['en'], gpu=False)

paddle_reader = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    # show_log=False
)

# -----------------------------
# Indian HSRP regex
# -----------------------------
INDIAN_PLATE_REGEX = re.compile(
    r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'
)

# Common OCR confusions
CHAR_CORRECTIONS = {
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'S': '5',
    'B': '8'
}

# -----------------------------
# Preprocessing for OCR
# -----------------------------
def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# -----------------------------
# Text normalization
# -----------------------------
def normalize_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)

    corrected = ''
    for i, ch in enumerate(text):
        if ch in CHAR_CORRECTIONS:
            corrected += CHAR_CORRECTIONS[ch]
        else:
            corrected += ch

    return corrected

# -----------------------------
# Plate validity check
# -----------------------------
def is_valid_plate(text):
    return bool(INDIAN_PLATE_REGEX.match(text))

# -----------------------------
# Ambiguity heuristic
# -----------------------------
def is_ambiguous(text):
    if text is None:
        return True
    if len(text) < 8 or len(text) > 11:
        return True
    if not is_valid_plate(text):
        return True
    return False

# -----------------------------
# EasyOCR
# -----------------------------
def ocr_easy(img):
    results = easy_reader.readtext(
        img,
        detail=0,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    if not results:
        return None
    return results[0]

# -----------------------------
# PaddleOCR
# -----------------------------
def ocr_paddle(img):
    results = paddle_reader.ocr(img, cls=True)
    if not results or not results[0]:
        return None

    texts = [line[1][0] for line in results[0]]
    return ''.join(texts)

# -----------------------------
# MAIN UTILITY FUNCTION
# -----------------------------
def anpr_ocr(image):
    """
    Args:
        image (np.ndarray): cropped number plate image (BGR)

    Returns:
        str | None: validated plate text
    """

    pre = preprocess_plate(image)

    # -------- EasyOCR first --------
    easy_text = ocr_easy(pre)
    if easy_text:
        easy_text = normalize_text(easy_text)
        if not is_ambiguous(easy_text):
            return easy_text

    # -------- PaddleOCR fallback --------
    paddle_text = ocr_paddle(pre)
    if paddle_text:
        paddle_text = normalize_text(paddle_text)
        if is_valid_plate(paddle_text):
            return paddle_text

    return None
