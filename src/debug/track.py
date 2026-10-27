import sqlite3, json, cv2
from pathlib import Path

VIDEO_ID = "video1.mp4"       # set to your video_id in DB
ROOT = Path(__file__).resolve().parents[2]
INPUT_VIDEO = ROOT / "data" / "videos" / "video1.mp4"
OUTPUT_VIDEO = ROOT / "output" / "track_101.mp4"
TRACK_ID = 14               # change to the track you want
CROP = False                 # set True to output only the crop

DB_PATH = ROOT / "data" / "database" / "detections.sqlite"
COLOR_VOCAB = ["black","blue","brown","green","grey","orange","pink","purple","red","white","yellow"]
COLOR_NAMES = ["footwear","hair","lowerbody","upperbody"]

def format_attrs(bin_probs, color_probs, topk=3):
    top = sorted(zip(range(len(bin_probs)), bin_probs), key=lambda x: x[1], reverse=True)[:topk]
    bin_str = ", ".join([f"b{idx}:{p:.2f}" for idx, p in top])
    color_strs = []
    for i, probs in enumerate(color_probs):
        if not probs: continue
        ci = max(range(len(probs)), key=lambda j: probs[j])
        color_strs.append(f"{COLOR_NAMES[i]}:{COLOR_VOCAB[ci]}")
    return bin_str + " | " + ", ".join(color_strs)

def main():
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT frame_idx, bbox_x1, bbox_y1, bbox_x2, bbox_y2, det_conf, bin_probs, color_probs "
        "FROM detections WHERE video_id=? AND track_id=? ORDER BY frame_idx",
        (VIDEO_ID, TRACK_ID)
    ).fetchall()
    conn.close()
    if not rows:
        print("No rows for that track.")
        return

    frame_map = {r[0]: r[1:] for r in rows}  # frame_idx -> (box..., det_conf, bin, color)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if CROP:
        # we’ll compute crop size on the fly per frame
        writer = None
    else:
        writer = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in frame_map:
            x1, y1, x2, y2, conf, bin_js, col_js = frame_map[idx]
            box = (int(x1), int(y1), int(x2), int(y2))
            bin_probs = json.loads(bin_js)
            color_probs = json.loads(col_js)
            label = f"ID {TRACK_ID} conf {conf:.2f} | " + format_attrs(bin_probs, color_probs)

            if CROP:
                x1c, y1c, x2c, y2c = box
                crop = frame[y1c:y2c, x1c:x2c]
                if writer is None:
                    ch, cw = crop.shape[:2]
                    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch))
                writer.write(crop)
            else:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (box[0], box[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                writer.write(frame)
        idx += 1

    cap.release()
    if writer: writer.release()
    print("Saved to", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()