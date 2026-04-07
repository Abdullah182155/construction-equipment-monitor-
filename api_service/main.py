"""
FastAPI Application — Main Entry Point.
Serves the REST API, WebSocket endpoints, and static frontend files.
"""

import asyncio
import logging
import os
import sys
import time
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
import shutil

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from api_service.routes import router, set_repository
from api_service.ws_manager import WebSocketManager

logger = logging.getLogger(__name__)

# Global instances
ws_manager = WebSocketManager()
_db = None
_repository = None
_cv_task: Optional[asyncio.Task] = None
_current_video_source: Optional[str] = None

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _init_database():
    """Initialize database connection and repository."""
    global _db, _repository
    try:
        from db_service.database import Database
        from db_service.repository import EquipmentRepository

        _db = Database(settings.database.url)
        _db.create_tables()
        _repository = EquipmentRepository(_db)
        set_repository(_repository)
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database not available (dashboard will use WebSocket only): {e}")


async def _run_cv_pipeline(video_source: Optional[str] = None):
    """Run CV pipeline in background and stream results via WebSocket."""
    try:
        from cv_service.pipeline import CVPipeline

        pipeline = CVPipeline()
        source = video_source or _current_video_source or settings.video.source

        logger.info(f"Starting CV pipeline with source: {source}")

        frame_interval = settings.api.ws_frame_interval
        jpeg_quality = settings.api.ws_jpeg_quality
        frame_count = 0

        for result in pipeline.process_video(source):
            frame_count += 1

            # Broadcast annotated frame via WebSocket
            if ws_manager.connection_count > 0:
                annotated_frame = result.get("annotated_frame")
                if annotated_frame is not None:
                    await ws_manager.broadcast_frame(annotated_frame, quality=jpeg_quality)

                # Broadcast equipment events
                for eq in result.get("equipment", []):
                    event_data = {
                        "equipment_id": eq["equipment_id"],
                        "timestamp": time.time(),
                        "state": eq["state"],
                        "activity": eq["activity"],
                        "active_time": round(eq["active_time"], 2),
                        "idle_time": round(eq["idle_time"], 2),
                        "utilization": round(eq["utilization"], 4),
                        "confidence": eq.get("activity_confidence", 0),
                        "algorithm": eq.get("algorithm", "rule_based"),
                        "motion_score": round(eq.get("motion_score", 0), 4),
                    }
                    await ws_manager.broadcast_event(event_data)

                    # Also write to database if available
                    if _repository:
                        try:
                            from datetime import datetime, timezone
                            event_data["timestamp"] = datetime.now(timezone.utc).isoformat()
                            _repository.insert_event(event_data)
                        except Exception as e:
                            logger.debug(f"DB write skipped: {e}")

                # Broadcast summary
                summary = result.get("summary", {})
                await ws_manager.broadcast_summary(summary)

            # Yield control to event loop
            await asyncio.sleep(frame_interval)

        logger.info(f"CV pipeline completed. Processed {frame_count} frames.")

    except asyncio.CancelledError:
        logger.info("CV pipeline cancelled")
    except Exception as e:
        logger.error(f"CV pipeline error: {e}", exc_info=True)
    finally:
        # Notify clients pipeline has stopped
        await ws_manager.broadcast_event({
            "type": "pipeline_stopped",
            "message": f"Pipeline finished. Processed {frame_count} frames."
        })


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )
    logger.info("=" * 60)
    logger.info("  Equipment Monitoring System — Starting Up")
    logger.info("=" * 60)

    _init_database()

    yield

    # Shutdown
    global _cv_task
    if _cv_task and not _cv_task.done():
        _cv_task.cancel()
    if _db:
        _db.close()
    logger.info("Application shut down")


# Create FastAPI app
app = FastAPI(
    title="Equipment Monitoring System",
    description="Real-time construction equipment monitoring via computer vision",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST API routes
app.include_router(router)

# Serve static frontend files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")


@app.get("/")
async def serve_frontend():
    """Serve the main dashboard HTML page."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend not found. API is running at /api/status"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time data streaming.
    
    Sends two types of messages:
    - type="frame": Base64-encoded JPEG video frame
    - type="event": Equipment state update JSON
    - type="summary": Aggregate metrics JSON
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()
            # Client can send commands (e.g., start/stop pipeline)
            try:
                msg = json.loads(data)
                if msg.get("command") == "start_pipeline":
                    global _cv_task
                    if _cv_task is None or _cv_task.done():
                        _cv_task = asyncio.create_task(_run_cv_pipeline())
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "data": {"message": "Pipeline started"}
                        }))
                elif msg.get("command") == "stop_pipeline":
                    if _cv_task and not _cv_task.done():
                        _cv_task.cancel()
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "data": {"message": "Pipeline stopped"}
                        }))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.disconnect(websocket)


@app.post("/api/pipeline/start")
async def start_pipeline():
    """Start the CV processing pipeline."""
    global _cv_task
    if _cv_task and not _cv_task.done():
        return {"status": "already_running"}
    _cv_task = asyncio.create_task(_run_cv_pipeline())
    return {"status": "started"}


@app.post("/api/pipeline/stop")
async def stop_pipeline():
    """Stop the CV processing pipeline."""
    global _cv_task
    if _cv_task and not _cv_task.done():
        _cv_task.cancel()
        return {"status": "stopped"}
    return {"status": "not_running"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for processing.
    Saves to data/uploads/ and sets it as the current video source.
    """
    global _current_video_source, _cv_task

    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        return {"status": "error", "message": f"Unsupported format: {suffix}. Use: {', '.join(allowed_extensions)}"}

    # Save file
    safe_name = f"upload_{int(time.time())}{suffix}"
    file_path = UPLOAD_DIR / safe_name
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return {"status": "error", "message": str(e)}

    _current_video_source = str(file_path)
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    logger.info(f"Video uploaded: {file.filename} ({file_size_mb:.1f} MB) → {file_path}")

    # Stop existing pipeline if running
    if _cv_task and not _cv_task.done():
        _cv_task.cancel()
        await asyncio.sleep(0.5)

    # Auto-start pipeline with uploaded video
    _cv_task = asyncio.create_task(_run_cv_pipeline(str(file_path)))

    return {
        "status": "success",
        "filename": file.filename,
        "size_mb": round(file_size_mb, 2),
        "message": "Video uploaded and pipeline started"
    }


@app.post("/api/download-url")
async def download_video_url(payload: dict):
    """
    Download a video from a URL (YouTube, direct link, etc.) using yt-dlp.
    Saves to data/uploads/ and starts pipeline automatically.
    """
    global _current_video_source, _cv_task

    url = payload.get("url", "").strip()
    if not url:
        return {"status": "error", "message": "No URL provided"}

    logger.info(f"Downloading video from: {url}")

    # Run yt-dlp in a thread to avoid blocking
    output_path = str(UPLOAD_DIR / f"yt_{int(time.time())}.mp4")

    def _download():
        try:
            import yt_dlp
            ydl_opts = {
                'format': 'best[ext=mp4][height<=720]/best[ext=mp4]/best',
                'outtmpl': output_path,
                'quiet': True,
                'no_warnings': True,
                'socket_timeout': 30,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return {
                    "title": info.get("title", "Unknown"),
                    "duration": info.get("duration", 0),
                }
        except Exception as e:
            raise e

    try:
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, _download)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return {"status": "error", "message": f"Download failed: {str(e)}"}

    # Verify file exists
    dl_path = Path(output_path)
    if not dl_path.exists():
        # yt-dlp may add extension, look for the file
        for f in UPLOAD_DIR.glob(f"yt_{dl_path.stem.split('_')[-1]}*"):
            dl_path = f
            break

    if not dl_path.exists():
        return {"status": "error", "message": "Download completed but file not found"}

    _current_video_source = str(dl_path)
    file_size_mb = dl_path.stat().st_size / (1024 * 1024)
    logger.info(f"Video downloaded: {info.get('title', 'Unknown')} ({file_size_mb:.1f} MB) → {dl_path}")

    # Stop existing pipeline if running
    if _cv_task and not _cv_task.done():
        _cv_task.cancel()
        await asyncio.sleep(0.5)

    # Auto-start pipeline
    _cv_task = asyncio.create_task(_run_cv_pipeline(str(dl_path)))

    return {
        "status": "success",
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0),
        "size_mb": round(file_size_mb, 2),
        "message": "Video downloaded and pipeline started"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_service.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
        log_level="info"
    )
