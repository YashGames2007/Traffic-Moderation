import threading
import queue
import time
import uuid
import cv2

from pipeline.anpr_by_model import detect_by_model
from pipeline.anpr_by_image_processing import detect_by_img_processing

# ===============================
# CONFIG
# ===============================
FRAME_QUEUE_SIZE = 1          # always keep latest frame
DETECTION_INTERVAL = 0.5      # seconds (2 FPS ANPR)
TOTAL_DEBUG_FRAMES = 5
TOTAL_OUTPUT_FRAMES = 5

# ===============================
# QUEUES
# ===============================
model_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
ip_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
result_queue = queue.Queue()

# ===============================
# MODEL WORKER
# ===============================
def model_worker():
    while True:
        frame, frame_id = model_queue.get()
        try:
            out_name = f"output/anpr_model_{frame_id}.jpg"
            plates = detect_by_model(frame, out_name)
            result_queue.put(("MODEL", frame_id, plates))
        except Exception as e:
            print("[MODEL ERROR]", e)

# ===============================
# IMAGE PROCESSING WORKER
# ===============================
def ip_worker():
    counter = 0
    while True:
        frame, frame_id = ip_queue.get()
        try:
            counter = ((counter + 1) % TOTAL_DEBUG_FRAMES)
            text = detect_by_img_processing(frame, output_dir="output", debug=True, debug_dir=f"output/ip{counter}", frame_id=frame_id)
            result_queue.put(("IMAGE_PROC", frame_id, text))
        except Exception as e:
            print("[IP ERROR]", e)

# ===============================
# RESULT AGGREGATOR
# ===============================
def result_consumer():
    while True:
        res = result_queue.get()
        if res[0] == "MODEL":
            _, frame_id, text = res
            if text != []:
                print(f"[MODEL] Frame {frame_id} OCR:", text)
        elif res[0] == "IMAGE_PROC":
            _, frame_id, text = res
            if text != None:
                print(f"[IMAGE_PROC] Frame {frame_id} OCR:", text)

# ===============================
# PIPELINE START
# ===============================
def start_pipeline():
    threading.Thread(target=model_worker, daemon=True).start()
    threading.Thread(target=ip_worker, daemon=True).start()
    threading.Thread(target=result_consumer, daemon=True).start()

# ===============================
# FRAME DISPATCHER
# ===============================
last_dispatch_time = 0
frame_counter = 0

def dispatch_frame(frame):
    global last_dispatch_time
    global frame_counter
    now = time.time()

    if now - last_dispatch_time < DETECTION_INTERVAL:
        return

    last_dispatch_time = now
    frame_id = frame_counter
    frame_counter = (frame_counter + 1) % TOTAL_OUTPUT_FRAMES

    # drop old frames automatically
    if not model_queue.full():
        model_queue.put((frame.copy(), frame_id))

    if not ip_queue.full():
        ip_queue.put((frame.copy(), frame_id))
