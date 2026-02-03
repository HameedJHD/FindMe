from ultralytics import YOLO

# Load YOLOv8n model (nano, lightweight) or change to yolov8s / yolov8m / yolov8l
model = YOLO("yolov8n.pt")

# Train
model.train(
    data="crowdhuman.yaml",
    epochs=50,            # adjust depending on your compute
    imgsz=640,            # input image size
    batch=8,              # batch size, adjust to your GPU
    device="cpu",             # GPU 0 (or 'cpu' if you don’t have GPU)
    project="runs/train", # output folder
    name="crowdhuman_yolo"
)
