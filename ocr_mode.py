import cv2
import pytesseract
import multiprocessing as mp
import threading
import queue
from picamera2 import Picamera2
from ultralytics import YOLO

CONF_THRESH = 0.4
IMG_SIZE = 640
FRAME_WIDTH = 1240
FRAME_HEIGHT = 840

# Load class names
with open("coco.names", "r") as f:
    class_names = [c.strip() for c in f.readlines()]

# Camera setup
picam2 = Picamera2()
picam2.preview_configuration.main.size = (FRAME_WIDTH, FRAME_HEIGHT)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# YOLO model
model = YOLO("yolov8n_openvino_model")

# TTS worker process
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

# Shared queue for frames
frame_queue = queue.Queue(maxsize=5)

# Camera thread → continuously push frames
def camera_thread():
    while True:
        frame = picam2.capture_array()
        if not frame_queue.full():
            frame_queue.put(frame)

# Processing thread → YOLO + OCR
def processing_thread():
    tts_queue.put("OCR mode on. Show me text in camera.")
    stop_flag = False
    while not stop_flag:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Object detection
            results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
            if results and results[0].boxes is not None:
                clss = results[0].boxes.cls.cpu().numpy().astype(int)
                bboxes = results[0].boxes.xyxy.cpu().numpy()
                for cls, box in zip(clss, bboxes):
                    x1, y1, x2, y2 = box.astype(int)
                    label = class_names[cls] if cls < len(class_names) else str(cls)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            if text.strip():
                cv2.putText(frame, f"Text: {text.strip()[:50]}...",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                tts_queue.put("I see text: " + text.strip())

            cv2.imshow("OCR + Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag = True
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    proc_thread = threading.Thread(target=processing_thread)

    cam_thread.start()
    proc_thread.start()

    proc_thread.join()  # wait until processing finishes
    tts_queue.put("STOP")
    picam2.stop()
