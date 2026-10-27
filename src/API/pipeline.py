import sys, time, cv2, os
from pathlib import Path
from ultralytics import YOLO
# Headless: disable Qt display usage
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT,):
    sp = str(p)
    if sp not in sys.path:
        sys.path.append(sp)

from src.db import DbClient
from src.attributes import AttrModel

DEVICE = 0
def _first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


DET_MODEL = _first_existing(
    ROOT / "models" / "yolo" / "yolov8m.pt",
    ROOT / "runs" / "weights" / "yolov8m.pt",
    ROOT / "yolov8m.pt",
)
ATTR_MODEL = _first_existing(
    ROOT / "models" / "peta" / "peta_resnet50_multihead.pt",
    ROOT / "peta_resnet50_multihead.pt",
)
VIDEO = ROOT / "data" / "videos" / "video1.mp4"
OUTPUT = ROOT / "output" / "output.mp4"
DB_PATH = ROOT / "data" / "database" / "detections.sqlite"
IMG_SIZE = 832
CONF_THRES = 0.25
IOU_THRES = 0.55
ATTR_EVERY_N_FRAMES = 1
TOPK_BIN = 3
COLOR_NAMES = ["footwear","hair","lowerbody","upperbody"]
COLOR_VOCAB = ["black","blue","brown","green","grey","orange","pink","purple","red","white","yellow"]

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

def format_attrs(attr_out):
    bins = attr_out["bin_probs"]
    colors = attr_out["color_probs"]
    topk = sorted(enumerate(bins), key=lambda x: x[1], reverse=True)[:TOPK_BIN]
    bin_str = ", ".join([f"b{idx}:{p:.2f}" for idx, p in topk])
    color_strs = []
    for i, probs in enumerate(colors):
        if not probs: continue
        ci = max(range(len(probs)), key=lambda j: probs[j])
        color_strs.append(f"{COLOR_NAMES[i]}:{COLOR_VOCAB[ci]}")
    return bin_str + " | " + ", ".join(color_strs)

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    detector = YOLO(str(DET_MODEL))
    attr_model = AttrModel(str(ATTR_MODEL), device=DEVICE)

    db = DbClient(DB_PATH)

    cap = cv2.VideoCapture(str(VIDEO))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        (w, h)
    )

    frame_idx = 0
    track_states = {}
    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = detector.track(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            classes=[0],
            persist=True,
            tracker="bytetrack.yaml"   # or "botsort.yaml" if you want
        )[0]

        # tracks is a list of [x1, y1, x2, y2, track_id, score]
        tracks = []
        for box, score, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.id.cpu().numpy()):
            tracks.append([*box, int(tid), float(score)])


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

        attr_map = {}
        if crops:
            attrs = attr_model.batch_predict(crops)
            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            for (tid, det_score, det_box), attr_out in zip(metas, attrs):
                track_states[tid]["last_attr_frame"] = frame_idx
                attr_map[tid] = attr_out
                db.insert(
                    video_id=VIDEO.name,
                    frame_idx=frame_idx,
                    timestamp_ms=ts_ms,
                    track_id=tid,
                    box=det_box,
                    det_conf=det_score,
                    attr_out=attr_out
                )

        # Draw overlays
        for t in tracks:
            x1, y1, x2, y2, tid, score = t
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            label = f"ID {tid} conf {score:.2f}"
            if tid in attr_map:
                label += " | " + format_attrs(attr_map[tid])
            cv2.putText(frame, label, (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        now = time.time()
        fps = 1.0 / max(1e-3, (now - prev_time))
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

if __name__ == "__main__":
    main()