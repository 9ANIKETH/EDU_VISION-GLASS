import cv2
import time
import threading
import queue
import multiprocessing as mp
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np

# ------------------- Config -------------------
CONF_THRESH = 0.4
IMG_SIZE = 640
FRAME_WIDTH = 640   # smaller for faster FPS
FRAME_HEIGHT = 480
FRAME_QUEUE_SIZE = 3

# Load COCO class names
with open("coco.names", "r") as f:
    class_names = [c.strip() for c in f.readlines()]

# ------------------- Camera Init -------------------
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam2.configure(preview_config)
picam2.start()

# ------------------- YOLOv8n Model -------------------
model = YOLO("yolov8n_openvino_model")  # or "yolov8n.pt"

# ------------------- TTS Worker -------------------
def tts_worker(queue):
    import pyttsx3
    engine = pyttsx3.init(driverName='espeak')
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    while True:
        text = queue.get()
        if text == "STOP":
            break
        engine.say(text)
        engine.runAndWait()

tts_queue = mp.Queue()
tts_process = mp.Process(target=tts_worker, args=(tts_queue,), daemon=True)
tts_process.start()

# ------------------- Frame Queue -------------------
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)

# ------------------- Camera Thread -------------------
def camera_thread():
    while True:
        frame = picam2.capture_array()
        if not frame_queue.full():
            frame_queue.put(frame)

# ------------------- Processing Thread -------------------
def processing_thread():
    stop_flag = False
    prev_time = time.time()
    while not stop_flag:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # YOLO object detection
            results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, cls_id in zip(boxes, cls_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Calculate and display FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow("YOLOv8n Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag = True
            break

    cv2.destroyAllWindows()

# ------------------- Main -------------------
if __name__ == "__main__":
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    proc_thread = threading.Thread(target=processing_thread)

    cam_thread.start()
    proc_thread.start()

    proc_thread.join()
    tts_queue.put("STOP")
    picam2.stop()
