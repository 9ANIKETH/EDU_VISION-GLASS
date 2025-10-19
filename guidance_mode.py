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
IOU_THRESH = 0.5
IMG_SIZE = 640
FRAME_WIDTH = 1240
FRAME_HEIGHT = 840
FRAME_QUEUE_SIZE = 3
DISTANCE_SCALE = 500
MAX_GUIDANCE_OBJECTS = 5
GUIDANCE_INTERVAL = 5  # seconds

# ------------------- Load Class Names -------------------
with open("coco.names", "r") as f:
    class_names = [c.strip() for c in f.readlines()]

# ------------------- Camera Init -------------------
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
)
picam2.configure(preview_config)
picam2.start()

# ------------------- YOLO Model -------------------
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

# ------------------- Processing & Guidance Thread -------------------
def processing_thread():
    latest_guidance = ""
    stop_flag = False
    annotated_frame = None
    lock = threading.Lock()
    prev_time = time.time()
    frame_count = 0

    def detect_objects():
        nonlocal latest_guidance, annotated_frame, stop_flag, prev_time, frame_count
        while not stop_flag:
            if not frame_queue.empty():
                frame = frame_queue.get()
                frame_count += 1

                # FPS calculation
                curr_time = time.time()
                fps = 1.0 / max((curr_time - prev_time), 1e-5)
                prev_time = curr_time

                # Track objects using YOLO
                results = model.track(
                    frame, imgsz=IMG_SIZE, conf=CONF_THRESH, iou=IOU_THRESH,
                    persist=True, tracker="bytetrack.yaml", verbose=False
                )

                guidance_messages = []

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else np.arange(len(boxes))

                    # Sort objects by width (largest first)
                    sorted_objects = sorted(zip(ids, cls_ids, boxes), key=lambda x: -(x[2][2]-x[2][0]))

                    for obj_id, cls, box in sorted_objects[:MAX_GUIDANCE_OBJECTS]:
                        x1, y1, x2, y2 = box.astype(int)
                        label = class_names[cls] if cls < len(class_names) else str(cls)
                        box_center_x = (x1 + x2) / 2

                        # Determine position for guidance
                        if box_center_x < FRAME_WIDTH / 3:
                            position, guidance = "Left", "move slightly to the right"
                        elif box_center_x > FRAME_WIDTH * 2 / 3:
                            position, guidance = "Right", "move slightly to the left"
                        else:
                            position, guidance = "Center", "stay in the center"

                        # Estimate distance
                        box_width = x2 - x1
                        distance = DISTANCE_SCALE / max(box_width, 1)
                        if distance < 0.5:
                            distance_str = "very close"
                        elif distance < 1.0:
                            distance_str = "1 hand distance"
                        elif distance < 1.5:
                            distance_str = "2 hand distance"
                        else:
                            distance_str = "far"

                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{obj_id} {label} ({position}, {distance_str})",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        guidance_messages.append(
                            f"{label} ID {obj_id} on your {position.lower()}, {distance_str}. Please {guidance}."
                        )

                if guidance_messages:
                    latest_guidance = " ".join(guidance_messages)

                # Overlay FPS
                cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                with lock:
                    annotated_frame = frame

    def tts_timer_worker():
        nonlocal latest_guidance, stop_flag
        while not stop_flag:
            time.sleep(GUIDANCE_INTERVAL)
            if latest_guidance:
                tts_queue.put(latest_guidance)

    threading.Thread(target=detect_objects, daemon=True).start()
    threading.Thread(target=tts_timer_worker, daemon=True).start()

    # Main display loop
    while True:
        with lock:
            if annotated_frame is not None:
                cv2.imshow("YOLOv8n Multi-Object Tracking", annotated_frame)
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
