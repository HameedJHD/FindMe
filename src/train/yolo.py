from ultralytics import YOLO

def main():
    # Load your best CrowdHuman-trained checkpoint
    model = YOLO("runs/weights/best.pt")

    # Fine-tune on COCO person-only
    model.train(
        data="config/coco.yaml",
        epochs=30,
        imgsz=640,
        batch=16,        # safe for 8GB VRAM
        workers=4,
        lr0=0.0005,      # LOW learning rate (important)
        freeze=10,       # freeze backbone
        device=0,
        amp=True,
        patience=10
        project="../../runs/train",
        name="yolo" 
    )

if __name__ == "__main__":
    main()