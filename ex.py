from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import time, threading, sys, subprocess

picam2 = Picamera2()

# 1080p @ 30fps (stable for Pi 5)
video_config = picam2.create_video_configuration(
    main={"size": (1920, 1080)},
    controls={"FrameRate": 30}
)
picam2.configure(video_config)

# Fullscreen preview
picam2.start_preview(Preview.QTGL, x=0, y=0, width=1920, height=1080)
picam2.start()

print("\n📷 Camera Preview Running (1080p @ 30fps, Fullscreen)...")
print("👉 Press 'r' to START recording")
print("👉 Press 's' to STOP recording (auto MP4 convert in background)")
print("👉 Press 'q' to QUIT\n")

recording = False
encoder = H264Encoder(10000000)  # 10 Mbps
video_file = ""
start_time = None
stop_timer = False

# Timer display thread
def show_timer():
    global start_time, stop_timer
    while not stop_timer:
        elapsed = int(time.time() - start_time)
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        sys.stdout.write(f"\r⏱ Recording... {hrs:02}:{mins:02}:{secs:02}")
        sys.stdout.flush()
        time.sleep(1)
    print()  # newline after stop

# Background MP4 conversion
def convert_to_mp4(h264_file):
    mp4_file = h264_file.replace(".h264", ".mp4")
    print(f"\n⏳ Converting {h264_file} -> {mp4_file} ...")
    subprocess.run(["ffmpeg", "-y", "-i", h264_file, "-c:v", "copy", mp4_file])
    print(f"🎬 Conversion done -> {mp4_file}")

while True:
    key = input("\nEnter option (r/s/q): ").strip().lower()

    if key == "r" and not recording:
        video_file = f"video_{int(time.time())}.h264"
        output = FileOutput(video_file)
        picam2.start_recording(encoder, output)

        recording = True
        start_time = time.time()
        stop_timer = False
        threading.Thread(target=show_timer, daemon=True).start()
        print(f"🎥 Recording started -> {video_file}")

    elif key == "s" and recording:
        stop_timer = True
        picam2.stop_recording()
        recording = False
        print(f"\n✅ Recording stopped. Saved -> {video_file}")

        # Start MP4 conversion in background
        threading.Thread(target=convert_to_mp4, args=(video_file,), daemon=True).start()

    elif key == "q":
        if recording:
            stop_timer = True
            picam2.stop_recording()
        picam2.stop_preview()
        picam2.close()
        print("👋 Exiting program...")
        break

    else:
        print("⚠️ Invalid input! Use: r = record, s = stop, q = quit")
