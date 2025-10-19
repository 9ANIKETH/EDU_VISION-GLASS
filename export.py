from ultralytics import YOLO

# Load PyTorch model
model = YOLO("yolo11n.pt")

# Export to NCNN format
model.export(format= "openvino")  # You can change the format and image size as needed
