# FindMe

FindMe is an intelligent video attribute search system. Its role is to turn
long, difficult-to-review videos into searchable records of people and their
visible characteristics.

Instead of requiring a user to watch an entire video manually, FindMe:

1. Accepts a video through a web interface.
2. Detects people in each frame with a YOLO model.
3. Tracks people across frames with ByteTrack.
4. Extracts visual attributes with a PETA-trained multi-head model.
5. Stores detections, track IDs, bounding boxes, and attribute probabilities
   in a per-video SQLite database.
6. Lets users search one or more processed videos using attribute and color
   requirements.
7. Produces a result video showing the matching people.

## What problem it solves

FindMe is designed for video investigation and retrieval tasks where a user
knows what a person looks like but does not know when or where that person
appears in a recording. Searches can combine binary attributes, such as
clothing or carried items, with colors for footwear, hair, lower-body
clothing, and upper-body clothing.

The system is an **attribute-based retrieval tool**, not a face-recognition
system. It describes detected people using model predictions and track
identifiers; it does not identify people by name.

## System architecture

```text
Browser
  │
  ▼
FastAPI backend (backend/app.py)
  │
  ├─ Upload and job management
  ├─ YOLO person detection
  ├─ ByteTrack tracking
  ├─ PETA attribute inference
  ├─ SQLite persistence
  └─ Attribute query and result-video rendering
```

### Main components

- `frontend/index.html` — browser user interface for login, uploads,
  processing progress, history, and searches.
- `backend/app.py` — primary FastAPI application and HTTP API.
- `backend/app1.py` — alternate backend implementation kept in the project
  for comparison or development.
- `src/API/pipeline.py` — video inference pipeline used by the backend.
- `src/attributes.py` — loads the attribute model and predicts binary and
  color probabilities for person crops.
- `src/reid.py` — optional FastReID feature extraction and track-linking
  support for stabilizing identities when tracks are fragmented.
- `src/db.py` — SQLite storage helper for frame-level detections and
  attributes.
- `src/peta_model.py` and `src/peta_dataset.py` — PETA attribute model and
  dataset support.
- `models/` — trained detection, attribute, and re-identification weights.
- `config/` — model and dataset configuration files.
- `vendor/fast-reid-1.3.0/` — vendored FastReID source used by the optional
  re-identification component.

## Typical workflow

### 1. Upload

The frontend sends a video to `POST /api/upload`. FindMe creates a job ID,
stores the source video under `data/uploads/<job_id>/`, and records metadata
for the job.

### 2. Process

The frontend starts processing with `POST /api/process/<job_id>`. The
pipeline analyzes the video frame by frame, detects people, assigns tracking
IDs, crops each person, and predicts their attributes.

### 3. Persist

Each job receives its own SQLite database under `data/database/`. Detection
records include:

- video and frame identifiers;
- timestamp and track ID;
- bounding-box coordinates;
- detector confidence;
- 61 binary attribute probabilities;
- four 11-class color probability distributions.

An annotated video is written under `output/`.

### 4. Search

After processing completes, the frontend submits binary and color
requirements to `POST /api/query`. FindMe checks the stored predictions
against configurable thresholds and renders a query-result video containing
matching tracks. `POST /api/query-multi` performs the same operation across
multiple completed jobs.

## API overview

All `/api/*` endpoints require a valid session token except `/api/login`.

| Endpoint | Purpose |
| --- | --- |
| `POST /api/login` | Create a session |
| `POST /api/upload` | Upload a source video |
| `POST /api/process/{job_id}` | Start video processing |
| `GET /api/status/{job_id}` | Read processing progress |
| `GET /api/jobs` | List completed jobs restored from disk |
| `POST /api/query` | Search one processed video |
| `POST /api/query-multi` | Search several processed videos |
| `GET /api/video/{job_id}/{vtype}` | Stream an output or query video |
| `POST /api/cancel/{job_id}` | Request cancellation of a running job |
| `DELETE /api/job/{job_id}` | Delete a job and its generated files |
| `POST /api/logout` | Invalidate the current session |

## Running the application

Install the API dependencies:

```bash
pip install -r requirements_api.txt
```

Install the machine-learning dependencies separately when running the
inference pipeline:

```bash
pip install -r requirements.txt
```

Start the FastAPI application from the project root:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser. The application expects the model
weights referenced by the pipeline to be available in the repository’s
`models/` directories.

## Generated data

The application creates these runtime directories as needed:

- `data/uploads/` — uploaded source videos grouped by job ID;
- `data/database/` — per-job SQLite detection databases;
- `output/` — annotated videos and search-result videos.

These files can be large and are runtime artifacts rather than source code.

## Important limitations

- Attribute predictions are probabilistic and depend on video quality,
  lighting, camera angle, and model performance.
- A tracking ID represents a track within the processed video; it is not a
  verified real-world identity.
- The pipeline is intended for authorized video analysis. Users should
  consider privacy, consent, retention, and applicable laws before processing
  video.
- GPU availability affects processing speed. The attribute model falls back
  to CPU when CUDA is unavailable.

## Training and experimentation

The repository also contains scripts and configuration for training or
evaluating the YOLO and PETA-based models. Those components support the
inference application but are separate from the main user-facing role of
FindMe: indexing videos so people can be retrieved by visual attributes.
