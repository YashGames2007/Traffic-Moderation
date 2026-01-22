import cv2
import numpy as np
from paddleocr import PaddleOCR

paddle_reader = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    # show_log=False
)

def ocr_paddle(img):
    if paddle_reader is None:
        return None

    try:
        results = paddle_reader.predict(img)
    except Exception as e:
        print("[PaddleOCR ERROR]", e)
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

def test_paddle_ocr():
    """
    Test PaddleOCR using img.jpg from project root.
    """

    img_path = "frame1.jpg"

    img = cv2.imread(img_path)

    if img is None:
        print("[TEST ERROR] Could not load img.jpg")
        return

    if not isinstance(img, np.ndarray):
        print("[TEST ERROR] Loaded object is not an image:", type(img))
        return

    print("[TEST] Image loaded:", img.shape, img.dtype)

    text = ocr_paddle(img)

    if text:
        print("[PADDLE OCR RESULT]:", text)
    else:
        print("[PADDLE OCR RESULT]: No text detected")


if __name__ == "__main__":
    test_paddle_ocr()
