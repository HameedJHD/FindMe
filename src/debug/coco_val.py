from ultralytics import YOLO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]

def main():
    # Load your fine-tuned COCO person-only checkpoint
    model_path = _first_existing(
        ROOT / "models" / "yolo" / "best.pt",
        ROOT / "runs" / "weights" / "best.pt",
    )
    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(ROOT / "config" / "coco.yaml"),
        imgsz=640,
        conf=0.001  # low threshold for proper mAP calculation
    )

    print(metrics)

if __name__ == "__main__":
    main()