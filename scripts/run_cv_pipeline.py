"""
Entrypoint: CV Pipeline + Kafka Producer
Runs the computer vision pipeline and publishes results to Kafka.
"""

import sys
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from cv_service.pipeline import CVPipeline
from kafka_service.producer import EquipmentProducer


def main():
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("cv_pipeline")

    logger.info("=" * 60)
    logger.info("  Equipment Monitoring — CV Pipeline + Kafka Producer")
    logger.info("=" * 60)
    logger.info(f"  Video Source: {settings.video.source}")
    logger.info(f"  YOLO Model:  {settings.yolo.model_path}")
    logger.info(f"  Motion Algo: {settings.motion.algorithm}")
    logger.info(f"  Activity:    {settings.motion.activity_algorithm}")
    logger.info(f"  Kafka:       {settings.kafka.bootstrap_servers}")
    logger.info("=" * 60)

    # Initialize
    pipeline = CVPipeline()
    producer = EquipmentProducer(settings.kafka)

    frame_count = 0
    events_sent = 0
    start_time = time.time()

    try:
        for result in pipeline.process_video(settings.video.source):
            frame_count += 1
            equipment = result.get("equipment", [])

            # Send events to Kafka
            sent = producer.send_batch(equipment)
            events_sent += sent

            # Log progress every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / max(elapsed, 0.001)
                logger.info(
                    f"Frame {frame_count} | "
                    f"{len(equipment)} equipment | "
                    f"{events_sent} events sent | "
                    f"{fps:.1f} FPS"
                )

                # Log per-equipment summary
                summary = result.get("summary", {})
                for eq_id, eq_data in summary.get("equipment", {}).items():
                    logger.info(
                        f"  Equipment #{eq_id}: "
                        f"{eq_data.get('state', 'N/A')} | "
                        f"{eq_data.get('activity', 'N/A')} | "
                        f"Util: {eq_data.get('utilization', 0):.0%}"
                    )

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    finally:
        producer.close()
        elapsed = time.time() - start_time
        logger.info(f"\nFinal Summary:")
        logger.info(f"  Total frames: {frame_count}")
        logger.info(f"  Total events: {events_sent}")
        logger.info(f"  Duration: {elapsed:.1f}s")
        logger.info(f"  Avg FPS: {frame_count / max(elapsed, 0.001):.1f}")


if __name__ == "__main__":
    main()
