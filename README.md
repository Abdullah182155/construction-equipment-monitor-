# 🏗️ Construction Equipment Monitoring System

> Real-time construction equipment monitoring via computer vision, activity classification, and utilization tracking.

A scalable microservices prototype that processes video streams of construction equipment (excavators), detects their state (ACTIVE/INACTIVE), classifies activities, computes utilization metrics, and streams results through Kafka to a premium web dashboard.

---

## 🏛️ Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────┐     ┌─────────────┐
│ Video Source │────▶│            CV Service                    │────▶│    Kafka     │
│ (file/RTSP/ │     │  ┌────────┐  ┌──────────┐  ┌─────────┐ │     │   Broker    │
│  webcam)    │     │  │YOLOv8n │─▶│ByteTrack │─▶│ Motion  │ │     │             │
│             │     │  │Detector│  │ Tracker  │  │Analyzer │ │     └──────┬──────┘
│             │     │  └────────┘  └──────────┘  └────┬────┘ │            │
│             │     │                                  │      │            │
│             │     │  ┌──────────────────────────────┐│      │     ┌──────▼──────┐
│             │     │  │  Activity Classifier         ││      │     │   Kafka     │
│             │     │  │  ┌───────┐ ┌────┐ ┌───┐     ││      │     │  Consumer   │
│             │     │  │  │ Rules │ │LSTM│ │C3D│     ││      │     └──────┬──────┘
│             │     │  │  └───────┘ └────┘ └───┘     ││      │            │
│             │     │  └──────────────────────────────┘│      │     ┌──────▼──────┐
│             │     │  ┌──────────────┐                │      │     │ PostgreSQL  │
│             │     │  │ Time Tracker │                │      │     │  Database   │
│             │     │  └──────────────┘                │      │     └──────┬──────┘
│             │     └──────────────────────────────────┘      │            │
│             │                        │                       │     ┌──────▼──────┐
│             │                        │ Annotated Frames      │     │  FastAPI    │
│             │                        │ via WebSocket          │     │  Backend    │
│             │                        │                       │     └──────┬──────┘
│             │                        ▼                       │            │
│             │              ┌──────────────────┐              │     ┌──────▼──────┐
│             │              │   Web Dashboard  │◀─────────────│─────│  Frontend   │
│             │              │  (HTML/CSS/JS)   │   REST + WS  │     │  Dashboard  │
│             │              └──────────────────┘              │     └─────────────┘
└─────────────┘                                                │
```

### Microservices

| Service | Description | Port |
|---------|-------------|------|
| **CV Service** | YOLOv8 detection + ByteTrack tracking + Motion analysis + Activity classification | - |
| **Kafka Producer** | Publishes equipment events to Kafka | - |
| **Kafka Consumer** | Reads events from Kafka, writes to PostgreSQL | - |
| **API Service** | FastAPI REST + WebSocket endpoints | 8000 |
| **Web Dashboard** | Premium dark-mode real-time dashboard | 8000 |

---

## 🧠 Computer Vision Pipeline

### Detection & Tracking
- **YOLOv8n** (nano) — Pretrained on COCO, filters to equipment classes
- **ByteTrack** via supervision — Persistent ID assignment across frames

### Motion Analysis (3 Algorithms)

| Algorithm | Purpose | Description |
|-----------|---------|-------------|
| **Optical Flow + YOLO** | State detection (ACTIVE/INACTIVE) | Dense Farneback optical flow computed within 3×3 bounding box grid. Handles articulated parts (arm moves while body is still). |
| **LSTM + YOLO** | Activity classification | Bidirectional LSTM (2 layers, 128 hidden) operating on 30-frame sequences of 27-dim feature vectors (kinematics + flow stats). |
| **C3D + YOLO** | Spatiotemporal features (optional) | 3D ConvNet on 16-frame clips cropped from bounding boxes. Best with GPU. |

### Recommended Pipeline
```
YOLO → ByteTrack → Optical Flow (state) → LSTM (activity) → Rule-based fallback
```

### Activity Classes
- **Digging** — Arm motion in upper regions, downward flow
- **Swinging/Loading** — Lateral motion, cab rotation pattern
- **Dumping** — Upper region motion with upward trajectory
- **Waiting** — No significant motion

### Articulated Motion Handling
The bounding box is split into a **3×3 grid** (9 zones). Each zone is analyzed independently for motion. This ensures:
- Arm-only motion is detected even when the cab/tracks are stationary
- Bucket movement triggers ACTIVE state
- Partial body rotation is captured

---

## 📦 Project Structure

```
├── config/
│   └── settings.py              # Centralized configuration (dataclasses + env vars)
├── cv_service/
│   ├── detector.py              # YOLOv8 object detection
│   ├── tracker.py               # ByteTrack multi-object tracking
│   ├── motion_analyzer.py       # Optical Flow + Frame Diff (strategy pattern)
│   ├── feature_extractor.py     # Feature vectors for LSTM/C3D
│   ├── lstm_model.py            # Bidirectional LSTM classifier
│   ├── c3d_model.py             # C3D spatiotemporal feature extractor
│   ├── activity_classifier.py   # Hybrid classifier (rules + LSTM + C3D)
│   ├── time_tracker.py          # Per-equipment utilization tracking
│   └── pipeline.py              # Main CV pipeline orchestrator
├── kafka_service/
│   ├── producer.py              # Kafka event producer
│   └── consumer.py              # Kafka event consumer
├── db_service/
│   ├── models.py                # SQLAlchemy ORM models
│   ├── database.py              # DB connection management
│   └── repository.py            # CRUD operations
├── api_service/
│   ├── main.py                  # FastAPI app + WebSocket
│   ├── routes.py                # REST API endpoints
│   └── ws_manager.py            # WebSocket connection manager
├── frontend/
│   ├── index.html               # Dashboard HTML
│   ├── css/styles.css           # Premium dark-mode styles
│   └── js/
│       ├── app.js               # Main app controller
│       ├── websocket.js         # WebSocket client
│       ├── charts.js            # Chart.js visualizations
│       └── video.js             # Video frame renderer
├── scripts/
│   ├── run_cv_pipeline.py       # CV + Kafka producer entrypoint
│   ├── run_consumer.py          # Kafka consumer + DB entrypoint
│   └── run_demo.py              # Full demo (recommended)
├── docker/
│   ├── Dockerfile.cv
│   ├── Dockerfile.consumer
│   └── Dockerfile.api
├── tests/
│   └── test_pipeline.py         # Pipeline unit tests
├── docker-compose.yml
├── requirements.txt
├── .env
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Local (Recommended for Development)

**Prerequisites:** Python 3.10+, pip

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place a video file (or change VIDEO_SOURCE in .env)
#    Download any excavator video and place at:
#    data/sample_video/excavator_demo.mp4

# 3. Run tests
python tests/test_pipeline.py

# 4. Start the demo server
python scripts/run_demo.py

# 5. Open dashboard
#    http://localhost:8000
#    Click "Start Pipeline" to begin processing
```

### Option 2: Docker Compose (Full Stack)

**Prerequisites:** Docker, Docker Compose

```bash
# 1. Place video file at data/sample_video/excavator_demo.mp4

# 2. Launch all services
docker-compose up --build

# 3. Open dashboard
#    http://localhost:8000
```

### Option 3: Individual Services

```bash
# Terminal 1: Start Kafka + PostgreSQL (via Docker)
docker-compose up zookeeper kafka postgres

# Terminal 2: CV Pipeline + Kafka Producer
python scripts/run_cv_pipeline.py

# Terminal 3: Kafka Consumer + DB Writer
python scripts/run_consumer.py

# Terminal 4: API + Dashboard
python scripts/run_demo.py
```

---

## ⚙️ Configuration

All settings are controlled via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_SOURCE` | `data/sample_video/excavator_demo.mp4` | Video file, RTSP URL, or webcam index |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model variant |
| `YOLO_CONFIDENCE` | `0.35` | Detection confidence threshold |
| `MOTION_THRESHOLD` | `2.5` | Motion detection sensitivity |
| `MOTION_ALGORITHM` | `optical_flow` | `optical_flow` or `frame_diff` |
| `ACTIVITY_ALGORITHM` | `hybrid` | `hybrid`, `lstm`, `c3d`, or `rule_based` |
| `LSTM_SEQUENCE_LENGTH` | `30` | LSTM input window (frames) |
| `TEMPORAL_SMOOTHING_FRAMES` | `5` | State smoothing window |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address |
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection string |

---

## 🌐 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/ws` | WebSocket | Real-time frames + events stream |
| `/api/status` | GET | System health check |
| `/api/equipment` | GET | List all tracked equipment |
| `/api/equipment/{id}/history` | GET | Event history for equipment |
| `/api/metrics` | GET | Aggregate utilization metrics |
| `/api/activities` | GET | Activity distribution counts |
| `/api/pipeline/start` | POST | Start CV pipeline |
| `/api/pipeline/stop` | POST | Stop CV pipeline |

### Kafka Message Format

```json
{
    "equipment_id": 1,
    "timestamp": "2026-04-06T14:30:00+00:00",
    "state": "ACTIVE",
    "activity": "Digging",
    "active_time": 120.50,
    "idle_time": 30.25,
    "utilization": 0.7993,
    "confidence": 0.85,
    "algorithm": "lstm",
    "motion_score": 3.4521,
    "bbox": [100, 200, 400, 500]
}
```

---

## 🎨 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **YOLOv8n** | Fastest YOLO variant, runs on CPU, sufficient for prototype |
| **ByteTrack** | No separate re-ID model needed, integrated via supervision library |
| **Optical Flow → State** | Dense motion estimation with region-based grid for articulated parts |
| **LSTM → Activity** | Captures temporal patterns in feature sequences, lightweight |
| **C3D → Optional** | Spatiotemporal learned features, best with GPU, swappable option |
| **Strategy pattern** | All motion analyzers share the same interface, swap via config |
| **Hybrid classifier** | Rules provide instant fallback; LSTM improves when loaded |
| **FastAPI + WebSocket** | Production-grade API, real-time streaming, serves static frontend |
| **Vanilla HTML/CSS/JS** | No build step, instant loading, premium glassmorphism design |
| **Chart.js** | Lightweight, beautiful charts, CDN loaded |
| **PostgreSQL** | Standard, reliable, supports time-series queries |
| **Docker Compose** | Single command deploys entire 6-service stack |

---

## 📊 Dashboard Features

- **Live video feed** with bounding boxes, state labels, and motion grid overlay
- **Equipment status cards** with real-time state, activity, time tracking
- **Utilization gauge** (fleet-wide)
- **Activity distribution** (doughnut chart)
- **Utilization timeline** (real-time line chart)
- **Algorithm status panel** showing active inference engines
- **Toast notifications** for connection and pipeline events
- **Auto-reconnecting WebSocket** with polling fallback
- **Responsive design** (works on desktop and tablet)

---

## 🔧 Extending the System

### Adding a New Activity Class
1. Add to `ACTIVITY_CLASSES` in `cv_service/activity_classifier.py`
2. Add heuristic rules in `RuleBasedClassifier.classify()`
3. Update LSTM `num_classes` in config
4. Retrain LSTM if using learned classification

### Using a Custom YOLO Model
1. Train YOLOv8 on construction equipment dataset
2. Set `YOLO_MODEL=path/to/custom.pt` in `.env`
3. Update `YOLO_TARGET_CLASSES` to match your class indices

### Switching Motion Algorithm
```bash
# In .env
MOTION_ALGORITHM=optical_flow   # or frame_diff
ACTIVITY_ALGORITHM=hybrid       # or lstm, c3d, rule_based
```

---

## 📜 License

This project is built as a technical assessment prototype. Not licensed for production use.

## Dashboard Preview
![Equipment Monitoring Dashboard](assets/dashboard.png)

## Dataset
The custom YOLO model was trained using the dataset available on Roboflow: [ACIDS Mini Dataset](https://universe.roboflow.com/emma-lin/acids-mini/dataset/3)
