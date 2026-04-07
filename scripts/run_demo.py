"""
Demo Script — Run the full system locally without Docker.
Starts the FastAPI server which handles CV pipeline, WebSocket streaming,
and serves the frontend dashboard.

Usage:
    python scripts/run_demo.py
    Then open http://localhost:8000 in your browser.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from config.settings import settings


def check_video_source():
    """Verify video source exists or provide instructions."""
    source = settings.video.source
    logger = logging.getLogger("demo")

    # Check if it's a file path
    if not source.startswith(("rtsp://", "http://")) and not source.isdigit():
        if not os.path.exists(source):
            logger.warning(f"Video file not found: {source}")
            logger.info("")
            logger.info("  To use the demo, place a video file at:")
            logger.info(f"    {os.path.abspath(source)}")
            logger.info("")
            logger.info("  Or set VIDEO_SOURCE in .env to:")
            logger.info("    - A video file path")
            logger.info("    - An RTSP URL (rtsp://...)")
            logger.info("    - A webcam index (0, 1, ...)")
            logger.info("")
            logger.info("  You can download a sample excavator video:")
            logger.info("    Search 'excavator working site' on pexels.com/videos")
            logger.info("")

            # Create sample directory
            os.makedirs(os.path.dirname(source) or ".", exist_ok=True)
            return False
    return True


def main():
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("demo")

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║   Equipment Monitoring System — Demo Mode               ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")
    logger.info("║                                                          ║")
    if settings.pipeline_mode == "custom":
        logger.info("║   Pipeline:  YOLO (best.pt) → Track → Utilization      ║")
        logger.info("║   Classes:   Excavator | Dump Truck | Concrete Mixer    ║")
    else:
        logger.info("║   Pipeline:  YOLO → Track → OptFlow → Classifier       ║")
    logger.info("║   Frontend:  http://localhost:8000                       ║")
    logger.info("║   API Docs:  http://localhost:8000/docs                  ║")
    logger.info("║                                                          ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("")

    video_ok = check_video_source()

    if video_ok:
        logger.info(f"Video source: {settings.video.source}")
    else:
        logger.info("Dashboard will start without video processing.")
        logger.info("Use the 'Start Pipeline' button after placing a video file.")

    logger.info(f"YOLO model: {settings.yolo.model_path}")
    logger.info(f"Pipeline mode: {settings.pipeline_mode}")
    logger.info("")
    logger.info("Starting server on http://localhost:8000 ...")
    logger.info("")

    import uvicorn
    uvicorn.run(
        "api_service.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
        log_level="info",
        access_log=False
    )


if __name__ == "__main__":
    main()
