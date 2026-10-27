"""
FindMe – Intelligent Video Attribute Search System
FastAPI Backend
"""

import sys, os, uuid, threading, shutil, json, sqlite3, cv2, secrets, queue, logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Directory setup ──────────────────────────────────────────────────────────
UPLOAD_DIR = ROOT / "data" / "uploads"
OUTPUT_DIR = ROOT / "output"
DB_DIR     = ROOT / "data" / "database"
for d in (UPLOAD_DIR, OUTPUT_DIR, DB_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGET_FPS = 24.0

logger = logging.getLogger("findme.api")

def _parse_allowed_origins() -> list[str]:
    raw = os.getenv("FINDME_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["http://localhost:3000", "http://127.0.0.1:3000"]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="FindMe API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require a valid session token for all /api/* routes except /api/login."""
    path = request.url.path
    # Allow preflight and login without a token.
    if (
        request.method == "OPTIONS"
        or path == "/api/login"
    ):
        return await call_next(request)
    if path.startswith("/api/"):
        # Support both X-Session-Token header and query-string token for media tags.
        token = request.headers.get("X-Session-Token", "") or request.query_params.get("token", "")
        if token not in sessions:
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    return await call_next(request)

# ── Authentication ────────────────────────────────────────────────────────────
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "123"
if not ADMIN_PASSWORD:
    raise RuntimeError(
        "FINDME_ADMIN_PASSWORD is required. "
        "Set it in your environment before starting the API."
    )
sessions: set  = set()   # valid session tokens (in-memory; cleared on restart)

# ── In-memory job registry ────────────────────────────────────────────────────
jobs: dict         = {}   # job_id -> status dict
cancel_flags: dict = {}   # job_id -> True when cancellation is requested

# ── Request models ────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    job_id: str
    bin_requirements: list[str] = []
    color_requirements: dict[str, str] = {}
    bin_thresh: float = 0.5
    color_thresh: float = 0.4

class MultiQueryRequest(BaseModel):
    job_ids: list[str]
    bin_requirements: list[str] = []
    color_requirements: dict[str, str] = {}
    bin_thresh: float = 0.5
    color_thresh: float = 0.4

# ── Attribute vocabulary (must match peta_dataset.py order exactly) ───────────
BINARY_ATTRS = [
    "accessoryHeadphone", "personalLess15",     "personalLess30",    "personalLess45",
    "personalLess60",     "personalLarger60",   "carryingBabyBuggy", "carryingBackpack",
    "hairBald",           "footwearBoots",      "lowerBodyCapri",    "carryingOther",
    "carryingShoppingTro","carryingUmbrella",   "lowerBodyCasual",   "upperBodyCasual",
    "personalFemale",     "carryingFolder",     "lowerBodyFormal",   "upperBodyFormal",
    "accessoryHairBand",  "accessoryHat",       "lowerBodyHotPants", "upperBodyJacket",
    "lowerBodyJeans",     "accessoryKerchief",  "footwearLeatherShoes","upperBodyLogo",
    "hairLong",           "lowerBodyLongSkirt", "upperBodyLongSleeve","lowerBodyPlaid",
    "lowerBodyThinStripes","carryingLuggageCase","personalMale",     "carryingMessengerBag",
    "accessoryMuffler",   "accessoryNothing",   "carryingNothing",   "upperBodyNoSleeve",
    "upperBodyPlaid",     "carryingPlasticBags","footwearSandals",   "footwearShoes",
    "hairShort",          "lowerBodyShorts",    "upperBodyShortSleeve","lowerBodyShortSkirt",
    "footwearSneaker",    "footwearStocking",   "upperBodyThinStripes","upperBodySuit",
    "carryingSuitcase",   "lowerBodySuits",     "accessorySunglasses","upperBodySweater",
    "upperBodyThickStripes","lowerBodyTrousers","upperBodyTshirt",   "upperBodyOther",
    "upperBodyVNeck",
]
BIN_IDX       = {k: i for i, k in enumerate(BINARY_ATTRS)}
COLOR_VOCAB   = ["black","blue","brown","green","grey","orange","pink","purple","red","white","yellow"]
COLOR_REGIONS = ["footwear","hair","lowerbody","upperbody"]
COLOR_RGN_IDX = {r: i for i, r in enumerate(COLOR_REGIONS)}
COLOR_VAL_IDX = {c: i for i, c in enumerate(COLOR_VOCAB)}

# ── Helper utilities ──────────────────────────────────────────────────────────
def _first(*paths) -> Path:
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return Path(paths[0])

def _crop_pad(frame, box, pad=0.05):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    pw, ph = pad * (x2 - x1), pad * (y2 - y1)
    x1 = max(0, int(x1 - pw));  y1 = max(0, int(y1 - ph))
    x2 = min(w, int(x2 + pw));  y2 = min(h, int(y2 + ph))
    return frame[y1:y2, x1:x2]

def _matches(bin_probs, color_probs, bin_reqs, color_reqs, bin_t, color_t):
    for name in bin_reqs:
        if name not in BIN_IDX:
            return False
        if bin_probs[BIN_IDX[name]] < bin_t:
            return False
    for region, color in color_reqs.items():
        ridx = COLOR_RGN_IDX.get(region)
        cidx = COLOR_VAL_IDX.get(color)
        if ridx is None or cidx is None:
            return False
        if color_probs[ridx][cidx] < color_t:
            return False
    return True

def _convert_to_fps(video_path: Path, target_fps: float) -> int:
    cap = cv2.VideoCapture(str(video_path))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_path = video_path.with_name(f"{video_path.stem}_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, target_fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to initialize VideoWriter for 24 fps conversion.")

    ratio = target_fps / max(fps_in, 1e-6)
    out_accum = 0.0
    out_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out_accum += ratio
        while out_accum >= 1.0:
            writer.write(frame)
            out_count += 1
            out_accum -= 1.0

    cap.release()
    writer.release()

    if out_count == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("No frames written during 24 fps conversion.")

    video_path.unlink(missing_ok=True)
    tmp_path.replace(video_path)
    return out_count

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(str(ROOT / "frontend" / "index.html"))

# -- Upload -------------------------------------------------------------------
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Save uploaded video, return job_id."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    video_path = job_dir / safe_name

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        total = _convert_to_fps(video_path, TARGET_FPS)
    except Exception as exc:
        raise HTTPException(500, f"Failed to convert video to {int(TARGET_FPS)} fps: {exc}")

    jobs[job_id] = {
        "status":       "uploaded",
        "video_path":   str(video_path),
        "video_name":   safe_name,
        "output_video": None,
        "db_path":      str(DB_DIR / f"{job_id}.sqlite"),
        "progress":     0,
        "total":        max(total, 1),
        "message":      "Video uploaded. Click Process to start.",
        "error":        None,
    }
    return {"job_id": job_id, "total_frames": total, "filename": safe_name}

# -- Start processing ---------------------------------------------------------
@app.post("/api/process/{job_id}")
async def start_processing(job_id: str):
    """Launch the AI pipeline in a daemon thread so it dies with the server."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    if jobs[job_id]["status"] == "processing":
        return {"message": "Already processing"}

    jobs[job_id]["status"]  = "processing"
    jobs[job_id]["message"] = "Starting pipeline…"
    t = threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True)
    t.start()
    return {"message": "Processing started", "job_id": job_id}

# -- Poll status --------------------------------------------------------------
@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    pct = round(j["progress"] / max(j["total"], 1) * 100, 1)
    return {**j, "percent": pct}

# -- Attribute query ----------------------------------------------------------
@app.post("/api/query")
async def query_attributes(req: QueryRequest):
    """Filter DB by attributes and render a result video."""
    if req.job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[req.job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Processing is not complete yet.")

    output_path = OUTPUT_DIR / f"query_{req.job_id}.mp4"
    result = _execute_query(
        db_path          = job["db_path"],
        video_path       = job["video_path"],
        output_path      = str(output_path),
        job_id           = req.job_id,
        bin_requirements = req.bin_requirements,
        color_requirements = req.color_requirements,
        bin_thresh       = req.bin_thresh,
        color_thresh     = req.color_thresh,
    )
    return result

# -- Serve video files --------------------------------------------------------
@app.get("/api/video/{job_id}/{vtype}")
async def serve_video(job_id: str, vtype: str):
    if vtype == "output":
        # Prefer the jobs dict, but fall back to filesystem so the endpoint
        # survives server restarts (the file exists even if the dict is gone).
        path = jobs.get(job_id, {}).get("output_video") or str(OUTPUT_DIR / f"output_{job_id}.mp4")
    elif vtype == "query":
        path = str(OUTPUT_DIR / f"query_{job_id}.mp4")
    else:
        raise HTTPException(400, "Unknown video type. Use 'output' or 'query'.")

    if not path or not Path(path).exists():
        raise HTTPException(404, "Video not ready yet.")

    # FileResponse handles Accept-Ranges / 206 Partial Content automatically.
    return FileResponse(
    path,
    media_type="video/mp4",
    headers={"Accept-Ranges": "bytes"}
)

# -- List all previously processed jobs --------------------------------------
@app.get("/api/jobs")
async def list_jobs():
    """Scan disk for every completed job and return its metadata.

    A job is considered complete when both its SQLite database AND its
    output video file exist.  The endpoint also restores each job's
    in-memory entry so that /api/query works immediately — even after a
    server restart — without any reprocessing.
    """
    result = []
    for db_file in sorted(DB_DIR.glob("*.sqlite"),
                          key=lambda p: p.stat().st_mtime, reverse=True):
        job_id    = db_file.stem
        out_video = OUTPUT_DIR / f"output_{job_id}.mp4"
        if not out_video.exists():
            continue  # pipeline never finished for this job

        # ── metadata from the database ──
        video_name  = job_id
        track_count = 0
        frame_count = 0
        try:
            conn = sqlite3.connect(str(db_file))
            row  = conn.execute(
                "SELECT video_id, COUNT(DISTINCT track_id), MAX(frame_idx) "
                "FROM detections"
            ).fetchone()
            conn.close()
            if row:
                if row[0]:
                    video_name  = row[0]
                track_count = int(row[1] or 0)
                frame_count = int((row[2] or 0) + 1)
        except Exception:
            logger.exception("Failed to restore metadata for job %s from %s", job_id, db_file)

        # ── restore in-memory state so searches work right away ──
        if job_id not in jobs:
            video_path = ""
            job_dir    = UPLOAD_DIR / job_id
            if job_dir.exists():
                for f in sorted(job_dir.iterdir()):
                    if f.is_file() and not f.name.startswith("."):
                        video_path = str(f)
                        break
            jobs[job_id] = {
                "status":       "done",
                "video_path":   video_path,
                "video_name":   video_name,
                "output_video": str(out_video),
                "db_path":      str(db_file),
                "progress":     frame_count,
                "total":        max(frame_count, 1),
                "message":      "Restored from disk.",
                "error":        None,
            }

        result.append({
            "job_id":       job_id,
            "video_name":   video_name,
            "track_count":  track_count,
            "frame_count":  frame_count,
            "processed_at": int(db_file.stat().st_mtime * 1000),  # epoch ms
        })

    return {"jobs": result}

# -- Delete a job and all its associated files ---------------------------------
@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Permanently remove every file associated with a job."""
    deleted = []
    missing = []

    # 1. Upload directory (contains the original source video)
    upload_dir = UPLOAD_DIR / job_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
        deleted.append(str(upload_dir))
    else:
        missing.append(str(upload_dir))

    # 2. SQLite database
    db_file = DB_DIR / f"{job_id}.sqlite"
    if db_file.exists():
        db_file.unlink()
        deleted.append(str(db_file))
    else:
        missing.append(str(db_file))

    # 3. Processed output video
    out_video = OUTPUT_DIR / f"output_{job_id}.mp4"
    if out_video.exists():
        out_video.unlink()
        deleted.append(str(out_video))
    else:
        missing.append(str(out_video))

    # 4. Search result video (only exists if a query was ever run)
    query_video = OUTPUT_DIR / f"query_{job_id}.mp4"
    if query_video.exists():
        query_video.unlink()
        deleted.append(str(query_video))

    # 5. Remove from in-memory registry
    jobs.pop(job_id, None)

    return {"deleted": deleted, "missing": missing}


# -- Bulk delete ---------------------------------------------------------------
class BulkDeleteRequest(BaseModel):
    job_ids: list[str]

@app.post("/api/jobs/delete")
async def bulk_delete_jobs(req: BulkDeleteRequest):
    """Delete multiple jobs in one request. Calls the same logic as single delete."""
    if not req.job_ids:
        raise HTTPException(400, "No job_ids provided.")
    if len(req.job_ids) > 50:
        raise HTTPException(400, "Maximum 50 jobs per bulk delete.")

    deleted = []
    failed  = []

    for job_id in req.job_ids:
        try:
            upload_dir = UPLOAD_DIR / job_id
            if upload_dir.exists():
                shutil.rmtree(upload_dir)

            for path in [
                DB_DIR     / f"{job_id}.sqlite",
                OUTPUT_DIR / f"output_{job_id}.mp4",
                OUTPUT_DIR / f"query_{job_id}.mp4",
            ]:
                if path.exists():
                    path.unlink()

            jobs.pop(job_id, None)
            deleted.append(job_id)
        except Exception as exc:
            failed.append({"job_id": job_id, "error": str(exc)})

    return {"deleted": deleted, "failed": failed}


# -- Multi-video attribute query -----------------------------------------------
@app.post("/api/query-multi")
async def query_attributes_multi(req: MultiQueryRequest):
    """Run the same attribute query across multiple jobs in sequence.

    Each job is queried against its own isolated database and produces its
    own result video.  /api/query (single-video) is not touched.
    """
    if not req.job_ids:
        raise HTTPException(400, "No job_ids provided.")
    if len(req.job_ids) > 20:
        raise HTTPException(400, "Maximum 20 videos per multi-search.")

    results = []
    for job_id in req.job_ids:
        if job_id not in jobs:
            results.append({
                "job_id":         job_id,
                "video_name":     job_id,
                "matched_tracks": 0,
                "video_url":      None,
                "message":        "Job not found — try reloading the history page.",
                "skipped":        True,
            })
            continue

        job = jobs[job_id]
        if job["status"] != "done":
            results.append({
                "job_id":         job_id,
                "video_name":     job.get("video_name", job_id),
                "matched_tracks": 0,
                "video_url":      None,
                "message":        "Processing not complete.",
                "skipped":        True,
            })
            continue

        output_path = OUTPUT_DIR / f"query_{job_id}.mp4"
        try:
            result = _execute_query(
                db_path            = job["db_path"],
                video_path         = job["video_path"],
                output_path        = str(output_path),
                job_id             = job_id,
                bin_requirements   = req.bin_requirements,
                color_requirements = req.color_requirements,
                bin_thresh         = req.bin_thresh,
                color_thresh       = req.color_thresh,
            )
        except Exception as exc:
            result = {
                "matched_tracks": 0,
                "video_url":      None,
                "message":        f"Error processing this video: {exc}",
            }

        result["job_id"]    = job_id
        result["video_name"] = job.get("video_name", job_id)
        result["skipped"]   = False
        results.append(result)

    total_matched = sum(r.get("matched_tracks", 0) for r in results)
    return {
        "results":         results,
        "total_matched":   total_matched,
        "videos_searched": len(req.job_ids),
    }

# ── Authentication endpoints ──────────────────────────────────────────────────

@app.post("/api/login")
async def login(req: LoginRequest):
    """Validate credentials and return a session token."""
    if req.username != ADMIN_USERNAME or req.password != ADMIN_PASSWORD:
        raise HTTPException(401, "Invalid username or password")
    token = secrets.token_hex(32)
    sessions.add(token)
    return {"token": token}

@app.post("/api/logout")
async def logout(request: Request):
    """Invalidate the caller's session token."""
    token = request.headers.get("X-Session-Token", "")
    sessions.discard(token)
    return {"message": "Logged out"}

@app.get("/api/auth/check")
async def auth_check():
    """Returns 200 if the caller's token is valid (middleware already checked)."""
    return {"authenticated": True}

# ── Cancel a running pipeline job ─────────────────────────────────────────────
@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Signal a running pipeline to stop at the next safe frame boundary."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    status = jobs[job_id].get("status")
    if status not in ("processing", "cancelling"):
        return {"message": f"Job is not in a cancellable state (status: {status})"}
    cancel_flags[job_id] = True
    jobs[job_id]["status"]  = "cancelling"
    jobs[job_id]["message"] = "Cancellation requested…"
    return {"message": "Cancellation requested"}

# Mount remaining static files (CSS, JS, images if any)
# IMPORTANT: this catch-all mount must be last — any route registered after it
# will be shadowed by the StaticFiles handler and will never execute.
app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="static")


# ── Shared file cleanup helper ────────────────────────────────────────────────
def _cleanup_job_files(job_id: str):
    """Remove every on-disk artifact for a job. Safe even if files are absent."""
    upload_dir = UPLOAD_DIR / job_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)
    for path in [
        DB_DIR     / f"{job_id}.sqlite",
        OUTPUT_DIR / f"output_{job_id}.mp4",
        OUTPUT_DIR / f"query_{job_id}.mp4",
    ]:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


# ── Pipeline background task ──────────────────────────────────────────────────

def _run_pipeline(job_id: str):
    """Runs YOLO + attribute model + DB insert in a background thread."""
    job = jobs[job_id]
    try:
        from ultralytics import YOLO
        from src.attributes import AttrModel
        from src.db import DbClient
        import torch

        DET_MODEL  = _first(ROOT / "models" / "yolo" / "best.pt")
        ATTR_MODEL = _first(ROOT / "models" / "peta" / "peta_resnet50_latestt.pt")

        jobs[job_id]["message"] = "Loading detection model…"
        detector   = YOLO(str(DET_MODEL))
        detector.fuse()

        torch.backends.cudnn.benchmark = True

        jobs[job_id]["message"] = "Loading attribute model…"
        attr_model = AttrModel(str(ATTR_MODEL), device=0)

        db         = DbClient(job["db_path"])
        video_path = Path(job["video_path"])
        out_path   = OUTPUT_DIR / f"output_{job_id}.mp4"

        cap    = cv2.VideoCapture(str(video_path))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_fps = 24.0
        writer_fps = min(fps_in, target_fps)
        # Prefer H.264 for browser playback; fall back to mp4v if encoder is unavailable.
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, (w, h))
        if not writer.isOpened():
            writer.release()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, (w, h))
        if not writer.isOpened():
            cap.release()
            db.close()
            raise RuntimeError("Failed to initialize output video writer.")

        frame_time = 1.0 / max(fps_in, 1e-6)
        infer_period = 1.0 / target_fps
        time_accum = 0.0
        frame_queue = queue.Queue(maxsize=32)
        writer_stop = object()

        def _writer_loop(q, w):
            while True:
                item = q.get()
                if item is writer_stop:
                    break
                w.write(item)

        writer_thread = threading.Thread(target=_writer_loop, args=(frame_queue, writer), daemon=True)
        writer_thread.start()

        status_every = 5
        frame_idx    = 0
        track_states = {}
        jobs[job_id]["message"] = "Processing frames…"

        while True:
            if cancel_flags.get(job_id):   # check before reading next frame
                break
            if fps_in > target_fps:
                time_accum += frame_time
                if time_accum < infer_period:
                    ok = cap.grab()
                    if not ok:
                        break
                    frame_idx += 1
                    jobs[job_id]["progress"] = frame_idx
                    if frame_idx % status_every == 0:
                        jobs[job_id]["message"]  = f"Frame {frame_idx} / {job['total']}"
                    continue
                time_accum -= infer_period

            ok, frame = cap.read()
            if not ok:
                break

            res = detector.track(
                source  = frame,
                imgsz   = 832,
                conf    = 0.25,
                iou     = 0.55,
                classes = [0],
                persist = True,
                tracker = "bytetrack.yaml",
            )[0]

            if cancel_flags.get(job_id):   # check after YOLO (inference can be slow)
                break

            # Build track list (guard against empty frames)
            tracks = []
            if res.boxes.id is not None:
                boxes = res.boxes.cpu()
                xyxy = boxes.xyxy.numpy()
                conf = boxes.conf.numpy()
                tids = boxes.id.numpy().astype(int)
                for box, score, tid in zip(xyxy, conf, tids):
                    tracks.append([*box, tid, float(score)])

            # Collect crops that need attribute inference
            crops, metas = [], []
            for x1, y1, x2, y2, tid, score in tracks:
                state = track_states.setdefault(tid, {"last_attr_frame": -999})
                if (frame_idx - state["last_attr_frame"]) >= 1:
                    crop = _crop_pad(frame, (x1, y1, x2, y2))
                    if crop.size > 0:
                        crops.append(crop)
                        metas.append((tid, score, (x1, y1, x2, y2)))

            # Run attribute model
            if crops:
                with torch.inference_mode():
                    attrs = attr_model.batch_predict(crops)
                ts_ms = frame_idx * 1000.0 / max(fps_in, 1e-6)
                for (tid, det_score, det_box), attr_out in zip(metas, attrs):
                    track_states[tid]["last_attr_frame"] = frame_idx
                    db.insert(
                        video_id     = video_path.name,
                        frame_idx    = frame_idx,
                        timestamp_ms = ts_ms,
                        track_id     = int(tid),
                        box          = det_box,
                        det_conf     = det_score,
                        attr_out     = attr_out,
                    )

            # Draw bounding boxes on annotated output video
            for x1, y1, x2, y2, tid, score in tracks:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"ID {int(tid)}  {score:.2f}",
                    (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                )

            frame_queue.put(frame)
            frame_idx += 1
            jobs[job_id]["progress"] = frame_idx
            if frame_idx % status_every == 0:
                jobs[job_id]["message"]  = f"Frame {frame_idx} / {job['total']}"

        cap.release()
        frame_queue.put(writer_stop)
        writer_thread.join()
        db.close()

        if cancel_flags.pop(job_id, False):
            # Cancelled — wipe every partial artifact so nothing is left behind
            _cleanup_job_files(job_id)
            jobs[job_id].update({
                "status":       "cancelled",
                "output_video": None,
                "message":      "Processing was cancelled.",
                "error":        None,
            })
        else:
            jobs[job_id].update({
                "status":       "done",
                "progress":     jobs[job_id]["total"],
                "output_video": str(out_path),
                "message":      "Processing complete!",
            })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        cancel_flags.pop(job_id, None)   # clean up flag if an exception fired first
        jobs[job_id].update({
            "status":  "error",
            "error":   str(exc),
            "message": f"Error: {exc}",
        })


# ── Attribute query engine ────────────────────────────────────────────────────

def _execute_query(db_path, video_path, output_path, job_id,
                   bin_requirements, color_requirements, bin_thresh, color_thresh):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT frame_idx, track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, "
        "det_conf, bin_probs, color_probs FROM detections"
    ).fetchall()
    conn.close()

    # Filter rows that match all requirements
    frame_map: dict = {}
    matched_ids: set = set()
    for frame_idx, tid, x1, y1, x2, y2, conf, bin_js, col_js in rows:
        bp = json.loads(bin_js)
        cp = json.loads(col_js)
        if _matches(bp, cp, bin_requirements, color_requirements, bin_thresh, color_thresh):
            matched_ids.add(tid)
            frame_map.setdefault(frame_idx, []).append((tid, (x1, y1, x2, y2), conf))

    if not frame_map:
        return {
            "matched_tracks": 0,
            "video_url": None,
            "message": "No matches found for these attributes. Try lowering the threshold.",
        }

    # Render result video — all frames, boxes only on matched persons
    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    target_fps = 24.0
    writer_fps = min(fps_in, target_fps)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, writer_fps, (w, h))

    frame_time = 1.0 / max(fps_in, 1e-6)
    infer_period = 1.0 / target_fps
    time_accum = 0.0

    src_idx = 0
    while True:
        if fps_in > target_fps:
            time_accum += frame_time
            if time_accum < infer_period:
                ok = cap.grab()
                if not ok:
                    break
                src_idx += 1
                continue
            time_accum -= infer_period

        ok, frame = cap.read()
        if not ok:
            break

        if src_idx in frame_map:
            for tid, box, conf in frame_map[src_idx]:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(
                    frame, f"ID {tid}  {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA,
                )
        writer.write(frame)
        src_idx += 1

    cap.release()
    writer.release()

    return {
        "matched_tracks": len(matched_ids),
        "video_url":      f"/api/video/{job_id}/query",
        "message":        f"Found {len(matched_ids)} matching person(s) across {len(frame_map)} frames.",
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
