import cv2
import time
from picamera2 import Picamera2

picam2 = Picamera2()

# Use video configuration (better for high FPS)
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)

# Force 60 FPS if supported
picam2.set_controls({"FrameRate": 60})

picam2.start()

prev_time = time.time()
frame_count = 0
fps = 0

while True:
    frame = picam2.capture_array()
    frame_count += 1
    current_time = time.time()

    if current_time - prev_time >= 1.0:
        fps = frame_count / (current_time - prev_time)
        prev_time = current_time
        frame_count = 0

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
