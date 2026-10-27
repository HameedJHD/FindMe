import cv2, torch, time
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.db import DbClient
from src.attributes import AttrModel

DEVICE = 0
def _first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


DET_MODEL = _first_existing(
    ROOT / "models" / "yolo" / "best.pt",
    ROOT / "runs" / "weights" / "best.pt",
    ROOT / "best.pt",
)
ATTR_MODEL = _first_existing(
    ROOT / "models" / "peta" / "peta_resnet50_multihead.pt",
    ROOT / "peta_resnet50_multihead.pt",
)
DB_PATH = ROOT / "data" / "database" / "detections.sqlite"
VIDEO_PATH = ROOT / "data" / "videos" / "video1.mp4"
CONF_THRES = 0.25
IOU_THRES = 0.55
IMG_SIZE = 832           # use 832 for speed; 960 for accuracy
ATTR_EVERY_N_FRAMES = 1  # run attributes every N frames per track

def crop_with_pad(frame, box, pad=0.05):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    pw, ph = pad * (x2 - x1), pad * (y2 - y1)
    x1 = max(0, int(x1 - pw)); y1 = max(0, int(y1 - ph))
    x2 = min(w, int(x2 + pw)); y2 = min(h, int(y2 + ph))
    return frame[y1:y2, x1:x2]

def should_run_attrs(track_state, frame_idx):
    last = track_state.get("last_attr_frame", -1)
    return (frame_idx - last) >= ATTR_EVERY_N_FRAMES

def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    detector = YOLO(str(DET_MODEL))
    attr_model = AttrModel(str(ATTR_MODEL), device=DEVICE)
    db = DbClient(DB_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    frame_idx = 0
    track_states = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = detector.track(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            classes=[0],
            persist=True,
            tracker="bytetrack.yaml"   # or "botsort.yaml" (better)
        )[0]

        tracks = []

        crops, metas = [], []
        for t in tracks:
            x1, y1, x2, y2, tid, score = t
            state = track_states.setdefault(tid, {"last_attr_frame": -999})
            if should_run_attrs(state, frame_idx):
                crop = crop_with_pad(frame, (x1, y1, x2, y2))
                if crop.size == 0:
                    continue
                crops.append(crop)
                metas.append((tid, score, (x1, y1, x2, y2)))

        if crops:
            attrs = attr_model.batch_predict(crops)
            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            for (tid, score, box), attr_out in zip(metas, attrs):
                track_states[tid]["last_attr_frame"] = frame_idx
                db.insert(
                    video_id=VIDEO_PATH.name,
                    frame_idx=frame_idx,
                    timestamp_ms=ts_ms,
                    track_id=tid,
                    box=box,
                    det_conf=score,
                    attr_out=attr_out
                )

        frame_idx += 1

    cap.release()
    db.close()

if __name__ == "__main__":
    main()