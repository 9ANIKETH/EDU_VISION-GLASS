import cv2
import time
from ultralytics import YOLO
import pyttsx3
import threading

# ---------------- CONFIG ----------------
CONF_THRESH = 0.4
IOU_THRESH = 0.5
IMG_SIZE = 640
DISTANCE_SCALE = 500
RISK_CONF_THRESH = 0.7

# ---------------- INIT TTS ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 1.0)

# ---------------- LOAD COCO NAMES ----------------
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# ---------------- INIT YOLO ----------------
model = YOLO("yolo11n_ncnn_model")  # Using NCNN model
print("[INFO] YOLO + ByteTrack Started on video. Press 'q' to quit.")

# ---------------- VIDEO ----------------
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video {video_path}")
    exit(1)

# ---------------- SHARED VARIABLES ----------------
frame = None
annotated_frame = None
lock = threading.Lock()
stop_flag = False

# ---------------- DETECTION THREAD ----------------
def detect_objects():
    global frame, annotated_frame, lock, stop_flag
    while not stop_flag:
        if frame is None:
            time.sleep(0.01)
            continue

        start_time = time.time()
        frame_copy = frame.copy()
        results = model.track(
            frame_copy,
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        if results and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            bboxes = results[0].boxes.xyxy.cpu().numpy()

            for obj_id, cls, conf, box in zip(ids, clss, confs, bboxes):
                x1, y1, x2, y2 = box.astype(int)
                label = class_names[cls] if cls < len(class_names) else str(cls)

                box_width = x2 - x1

                # Calculate overlap with Left / Center / Right regions
                left_overlap = max(0, min(x2, frame_copy.shape[1]/3) - x1) / box_width
                right_overlap = max(0, x2 - max(x1, frame_copy.shape[1]*2/3)) / box_width
                center_overlap = max(0, min(x2, frame_copy.shape[1]*2/3) - max(x1, frame_copy.shape[1]/3)) / box_width

                if left_overlap >= 0.6:
                    position = "Left"
                    direction_alert = "You go to the right"
                elif right_overlap >= 0.6:
                    position = "Right"
                    direction_alert = "You go to the left"
                elif center_overlap >= 0.6:
                    position = "Center"
                    direction_alert = "Stay in the center"
                else:
                    position = "Center"
                    direction_alert = "Stay in the center"

                # Estimate relative distance
                distance = DISTANCE_SCALE / max(box_width, 1)

                # Convert distance to hand-based verbalization
                if distance < 0.5:
                    distance_str = "very close"
                elif distance < 1.0:
                    distance_str = "1 hand distance"
                elif distance < 1.5:
                    distance_str = "2 hand distance"
                else:
                    distance_str = "far"

                # Draw bounding box and text
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_copy,
                    f"ID {obj_id}: {label} ({position}) Dist: {distance_str}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Risk alert + TTS
                if conf >= RISK_CONF_THRESH:
                    alert_text = f"[RISK ALERT] ID {obj_id}: {label}, Position: {position}, Conf: {conf:.2f}, Dist: {distance_str}"
                    print(alert_text)
                    speech_text = f"{label} detected on your {position}, {distance_str}. {direction_alert}."
                    threading.Thread(target=lambda: engine.say(speech_text) or engine.runAndWait()).start()

        # FPS
        fps = 1 / (time.time() - start_time + 1e-6)
        cv2.putText(frame_copy, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        with lock:
            annotated_frame = frame_copy

# Start detection thread
threading.Thread(target=detect_objects, daemon=True).start()

# ---------------- MAIN LOOP ----------------
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video ended or cannot read frame.")
            break

        with lock:
            if annotated_frame is not None:
                cv2.imshow("ByteTrack + FPS (Video)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag = True
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Video processing ended.")
