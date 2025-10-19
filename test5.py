from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLOv8n model (pretrained or OpenVINO optimized)
model = YOLO("yolo11n_openvino_model")  # or "yolov8n.pt"

# Initialize Picamera2
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(preview_config)
picam2.start()

try:
    while True:
        # Capture frame from camera
        frame = picam2.capture_array()  # NumPy array (RGB)
        
        # YOLO expects BGR images, convert if needed
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run YOLO detection
        results = model(frame_bgr)
        
        # Draw boxes and labels
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = f"{cls_id}:{conf:.2f}"
                
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame_bgr, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        # Show frame
        cv2.imshow("YOLOv8n Live Detection (PiCamera2)", frame_bgr)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
