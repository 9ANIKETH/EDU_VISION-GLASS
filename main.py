import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
import threading
import multiprocessing as mp
import RPi.GPIO as GPIO
import speech_recognition as sr
import google.generativeai as genai
import os
import pytesseract   # for OCR

# ---------------- CONFIG ----------------
CONF_THRESH = 0.4
IOU_THRESH = 0.5
IMG_SIZE = 640
FRAME_WIDTH = 1240
FRAME_HEIGHT = 840
DISTANCE_SCALE = 500
MAX_GUIDANCE_OBJECTS = 5
GUIDANCE_INTERVAL = 5
BUTTON_PIN = 17

# Disable unnecessary logs
os.environ["YOLO_VERBOSE"] = "False"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# ---------------- LOAD COCO NAMES ----------------
with open("coco.names", "r") as f:
    class_names = [c.strip() for c in f.readlines()]

# ---------------- INIT CAMERA ----------------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (FRAME_WIDTH, FRAME_HEIGHT)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# ---------------- INIT YOLO ----------------
model = YOLO("yolov8n_openvino_model")

# ---------------- GEMINI CONFIG ----------------
API_KEY = "AIzaSyCHHv141Mi3O6sCr_Jgq3ysR6-CIh2-9z0"
genai.configure(api_key=API_KEY)
model_ai = genai.GenerativeModel('gemini-1.5-flash')
convo = model_ai.start_chat()
convo.send_message("You are a short and clear voice assistant.")

recognizer = sr.Recognizer()
stop_word = "exit"

# ---------------- MULTIPROCESS TTS ----------------
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

# ---------------- SPEECH RECOGNITION ----------------
def listen_microphone(timeout=5):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=timeout)
            return recognizer.recognize_google(audio).lower()
        except:
            return ""

# ---------------- CONVERSATION MODE ----------------
def conversation_loop():
    global tts_queue   # ✅ यह line जोड़ना जरूरी है
    tts_queue.put("Conversation mode on. Say something.")

    while True:
        tts_queue.put("Listening...")
        user_input = listen_microphone()
        
        if not user_input:
            continue

        if stop_word in user_input:
            tts_queue.put("Exiting conversation mode")
            break

        # Send to Gemini
        convo.send_message(user_input)
        response = convo.last.text

        if response:
            tts_queue.put(response)


# ---------------- GUIDANCE DETECTION MODE ----------------
def guidance_mode():
    latest_guidance = ""
    stop_flag = False
    annotated_frame = None
    lock = threading.Lock()

    def detect_objects():
        nonlocal latest_guidance, annotated_frame, stop_flag
        while not stop_flag:
            fcopy = picam2.capture_array()
            results = model.track(fcopy, imgsz=IMG_SIZE, conf=CONF_THRESH, iou=IOU_THRESH,
                                  persist=True, tracker="bytetrack.yaml", verbose=False)

            guidance_messages = []

            if results and results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)
                bboxes = results[0].boxes.xyxy.cpu().numpy()

                sorted_objects = sorted(zip(ids, clss, bboxes), key=lambda x: -(x[2][2]-x[2][0]))

                for i, (obj_id, cls, box) in enumerate(sorted_objects[:MAX_GUIDANCE_OBJECTS]):
                    x1, y1, x2, y2 = box.astype(int)
                    label = class_names[cls] if cls < len(class_names) else str(cls)
                    box_center_x = (x1 + x2) / 2

                    if box_center_x < FRAME_WIDTH / 3:
                        position, guidance = "Left", "move slightly to the right"
                    elif box_center_x > FRAME_WIDTH * 2 / 3:
                        position, guidance = "Right", "move slightly to the left"
                    else:
                        position, guidance = "Center", "stay in the center"

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

                    cv2.rectangle(fcopy, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(fcopy, f"{label} ({position}, {distance_str})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    guidance_messages.append(f"{label} on your {position.lower()}, {distance_str}. Please {guidance}.")

            if guidance_messages:
                latest_guidance = " ".join(guidance_messages)

            with lock:
                annotated_frame = fcopy

    def tts_timer_worker():
        nonlocal latest_guidance, stop_flag
        while not stop_flag:
            time.sleep(GUIDANCE_INTERVAL)
            if latest_guidance:
                tts_queue.put(latest_guidance)

    threading.Thread(target=detect_objects, daemon=True).start()
    threading.Thread(target=tts_timer_worker, daemon=True).start()

    while True:
        with lock:
            if annotated_frame is not None:
                cv2.imshow("Guidance Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag = True
            break
    cv2.destroyAllWindows()

# ---------------- OCR MODE ----------------
# ---------------- OCR MODE (Live Preview + Object + Text) ----------------
def ocr_mode():
    tts_queue.put("OCR mode on. Show me text in camera.")
    stop_flag = False

    while not stop_flag:
        frame = picam2.capture_array()

        # ---------------- OBJECT DETECTION ----------------
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

        # ---------------- OCR DETECTION ----------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)

        if text.strip():
            cv2.putText(frame, f"Text: {text.strip()[:50]}...",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            tts_queue.put("I see text: " + text.strip())

        # ---------------- PREVIEW ----------------
        cv2.imshow("OCR + Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag = True
            break

    cv2.destroyAllWindows()


# ---------------- MAIN MENU ----------------
while True:
    print("\n===== MAIN MENU =====")
    print("1. Guidance Detection Mode")
    print("2. AI Conversation Mode")
    print("3. OCR Text Read Mode")
    print("4. Exit")
    choice = input("Select an option: ")

    if choice == "1":
        guidance_mode()
    elif choice == "2":
        conversation_loop()
    elif choice == "3":
        ocr_mode()
    elif choice == "4":
        tts_queue.put("STOP")
        picam2.stop()
        GPIO.cleanup()
        break
    else:
        print("Invalid choice! Please select 1-4.")
