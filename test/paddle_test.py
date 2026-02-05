# import cv2
# print(cv2._version_)
# import numpy
# import cv2
# from paddleocr import PaddleOCR

# print("NumPy:", numpy._version_)
# print("OpenCV:", cv2._version_)
# print("PaddleOCR ready")


from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(
    lang='en',
    use_angle_cls=False,
    enable_mkldnn=False,
    show_log=False
)

img = cv2.imread("frame1.jpg")
results = ocr.ocr(img)

S=''
for result in results:
    for line in result:
        text = line[0]
        confidence = line[1]
        if isinstance(text, str):
            S+=text
        # print(text, "|", confidence)
print(S)