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
    'O': '0', 
    'Q': '0', 
    'D': '0',
    'G': '6',
    'I': '1', 
    'L': '1',
    'Z': '2',
    'S': '5',
    'B': '8'
}

DIGIT_CORRECTIONS = {
    '0': 'O', 
    '0': 'Q', 
    '0': 'D',
    '6': 'G',
    '1': 'I', 
    '1': 'L',
    '2': 'Z',
    '5': 'S',
    '8': 'B'
}

# -----------------------------
# Preprocessing for OCR
# -----------------------------
def preprocess_plate(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh

# -----------------------------
# Text normalization
# -----------------------------
def normalize_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    print(f"Before Correction: {text}")
    corrected = ''
    for i, ch in enumerate(text):
        if i in [2, 3, 6, 7, 8, 9] and ch in CHAR_CORRECTIONS: # Digit Places in Number Plate
                corrected += CHAR_CORRECTIONS[ch]
        elif i not in [2, 3, 6, 7, 8, 9] and ch in DIGIT_CORRECTIONS:
                corrected += DIGIT_CORRECTIONS[ch]
        else:
            corrected += ch
    print(f"After Correction: {corrected}")
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
        return False
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
    if paddle_reader is None:
        return None

    try:
        results = paddle_reader.predict(img)
    except Exception as e:
        # print("[PaddleOCR ERROR]", e)
        return None

    if not results or not isinstance(results, list):
        return None

    texts = []

    for block in results:
        if not block:
            continue

        for line in block:
            # line must be: [bbox, (text, confidence)]
            if (
                isinstance(line, (list, tuple)) and
                len(line) >= 2 and
                isinstance(line[1], (list, tuple)) and
                len(line[1]) >= 1
            ):
                text = line[1][0]
                if isinstance(text, str):
                    texts.append(text)

    if not texts:
        return None

    return "".join(texts)

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

    pre = image

    # -------- EasyOCR first --------
    easy_text = ocr_easy(pre)
    if easy_text:
        easy_text = normalize_text(easy_text)
        if not is_ambiguous(easy_text):
            return easy_text
    if easy_text is None:
        return easy_text  # Early Exit if no Numbers are detected
    # print(f"Ambiguous Text by Easy OCR {easy_text}")
    # -------- PaddleOCR fallback -------- (IF Inaccurate detection happened)
    paddle_text = ocr_paddle(pre)
    if paddle_text:
        paddle_text = normalize_text(paddle_text)
        if is_valid_plate(paddle_text):
            return paddle_text

    if is_valid_plate(easy_text):
        return easy_text
    return None
