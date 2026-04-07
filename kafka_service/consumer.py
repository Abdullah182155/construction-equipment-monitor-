"""
Kafka Consumer for Equipment Events.
Subscribes to Kafka topic and writes events to PostgreSQL.
"""

import json
import logging
import signal
import sys
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class EquipmentConsumer:
    """
    Kafka consumer that reads equipment events and stores them in the database.
    
    Handles:
        - Auto-reconnection
        - Graceful shutdown
        - Batch processing for efficiency
    """

    def __init__(self, config, repository=None):
        """
        Args:
            config: KafkaConfig
            repository: DBRepository instance for writing to database
        """
        self.config = config
        self.repository = repository
        self.consumer = None
        self.running = False
        self._setup_signal_handlers()

    def _connect(self):
        """Connect to Kafka broker."""
        try:
            from kafka import KafkaConsumer

            self.consumer = KafkaConsumer(
                self.config.topic,
                bootstrap_servers=self.config.bootstrap_servers,
                group_id=self.config.group_id,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                consumer_timeout_ms=1000,
                max_poll_interval_ms=300000,
            )
            logger.info(
                f"Kafka consumer connected to {self.config.bootstrap_servers}, "
                f"topic={self.config.topic}, group={self.config.group_id}"
            )
        except Exception as e:
            logger.warning(f"Kafka consumer connection failed: {e}")
            self.consumer = None

    def _setup_signal_handlers(self):
        """Setup graceful shutdown on SIGINT/SIGTERM."""
        def handler(signum, frame):
            logger.info("Shutdown signal received, stopping consumer...")
            self.running = False

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def start(self, callback: Optional[Callable] = None):
        """
        Start consuming messages in a loop.

        Args:
            callback: Optional function to call with each message dict.
                      If None and repository is set, writes to database.
        """
        self.running = True
        retry_delay = 5

        while self.running:
            if self.consumer is None:
                self._connect()
                if self.consumer is None:
                    logger.info(f"Retrying Kafka connection in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60)
                    continue
                retry_delay = 5

            try:
                for message in self.consumer:
                    if not self.running:
                        break

                    event = message.value
                    logger.debug(
                        f"Consumed event: equipment_id={event.get('equipment_id')}, "
                        f"state={event.get('state')}, activity={event.get('activity')}"
                    )

                    # Process event
                    if callback:
                        callback(event)
                    elif self.repository:
                        try:
                            self.repository.insert_event(event)
                        except Exception as e:
                            logger.error(f"Failed to write event to DB: {e}")

            except Exception as e:
                logger.error(f"Consumer error: {e}")
                self.consumer = None
                time.sleep(retry_delay)

        self.close()

    def close(self):
        """Close the consumer connection."""
        if self.consumer:
            try:
                self.consumer.close()
                logger.info("Kafka consumer closed")
            except Exception:
                pass
