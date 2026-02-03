from ultralytics import YOLOv8 as yolo
# Example command to train YOLOv8 on CrowdHuman
model = YOLO("yolov8n.pt")
results = model("test.jpg", show=True)
