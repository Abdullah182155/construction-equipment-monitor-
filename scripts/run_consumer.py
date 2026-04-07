"""
Entrypoint: Kafka Consumer + Database Writer
Consumes equipment events from Kafka and stores them in PostgreSQL.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from db_service.database import Database
from db_service.repository import EquipmentRepository
from kafka_service.consumer import EquipmentConsumer


def main():
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("consumer")

    logger.info("=" * 60)
    logger.info("  Equipment Monitoring — Kafka Consumer + DB Writer")
    logger.info("=" * 60)
    logger.info(f"  Kafka:    {settings.kafka.bootstrap_servers}")
    logger.info(f"  Topic:    {settings.kafka.topic}")
    logger.info(f"  Group:    {settings.kafka.group_id}")
    logger.info(f"  Database: {settings.database.url.split('@')[-1] if '@' in settings.database.url else 'local'}")
    logger.info("=" * 60)

    # Initialize database
    db = Database(settings.database.url)
    db.create_tables()
    repository = EquipmentRepository(db)

    # Initialize and start consumer
    consumer = EquipmentConsumer(settings.kafka, repository)

    event_count = 0

    def on_event(event):
        nonlocal event_count
        event_count += 1
        repository.insert_event(event)
        if event_count % 50 == 0:
            logger.info(f"Processed {event_count} events")

    try:
        logger.info("Starting Kafka consumer... (Press Ctrl+C to stop)")
        consumer.start(callback=on_event)
    except KeyboardInterrupt:
        logger.info("Consumer interrupted")
    finally:
        consumer.close()
        db.close()
        logger.info(f"Total events processed: {event_count}")


if __name__ == "__main__":
    main()
