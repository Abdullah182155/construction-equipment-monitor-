"""
Kafka Producer for Equipment Events.
Serializes CV pipeline results to JSON and publishes to Kafka topic.
"""

import json
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class EquipmentProducer:
    """
    Kafka producer that publishes equipment monitoring events.
    
    Message format:
    {
        "equipment_id": 1,
        "timestamp": "2026-04-06T14:30:00Z",
        "state": "ACTIVE",
        "activity": "Digging",
        "active_time": 120.5,
        "idle_time": 30.2,
        "utilization": 0.80,
        "confidence": 0.85,
        "algorithm": "lstm",
        "motion_score": 3.45,
        "bbox": [100, 200, 400, 500]
    }
    """

    def __init__(self, config):
        self.config = config
        self.producer = None
        self._connect()

    def _connect(self):
        """Connect to Kafka broker."""
        try:
            from kafka import KafkaProducer

            self.producer = KafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8") if k else None,
                acks="all",
                retries=3,
                max_block_ms=5000,
            )
            logger.info(f"Kafka producer connected to {self.config.bootstrap_servers}")
        except Exception as e:
            logger.warning(f"Kafka connection failed (will retry): {e}")
            self.producer = None

    def send_event(self, equipment_result: Dict[str, Any]) -> bool:
        """
        Publish a single equipment event to Kafka.

        Args:
            equipment_result: Dict from CV pipeline with equipment data

        Returns:
            True if sent successfully, False otherwise
        """
        if self.producer is None:
            self._connect()
            if self.producer is None:
                return False

        try:
            message = {
                "equipment_id": equipment_result.get("equipment_id", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "state": equipment_result.get("state", "INACTIVE"),
                "activity": equipment_result.get("activity", "Waiting"),
                "active_time": round(equipment_result.get("active_time", 0), 2),
                "idle_time": round(equipment_result.get("idle_time", 0), 2),
                "utilization": round(equipment_result.get("utilization", 0), 4),
                "confidence": equipment_result.get("activity_confidence", 0),
                "algorithm": equipment_result.get("algorithm", "rule_based"),
                "motion_score": round(equipment_result.get("motion_score", 0), 4),
                "bbox": equipment_result.get("bbox", []),
            }

            key = str(message["equipment_id"])
            self.producer.send(
                self.config.topic,
                key=key,
                value=message
            )
            logger.debug(f"Event sent for equipment #{message['equipment_id']}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Kafka event: {e}")
            return False

    def send_batch(self, equipment_results: list) -> int:
        """Send a batch of equipment events. Returns count of successful sends."""
        sent = 0
        for result in equipment_results:
            if self.send_event(result):
                sent += 1
        self.flush()
        return sent

    def flush(self):
        """Flush all pending messages."""
        if self.producer:
            try:
                self.producer.flush(timeout=5)
            except Exception as e:
                logger.warning(f"Flush failed: {e}")

    def close(self):
        """Close the producer connection."""
        if self.producer:
            try:
                self.producer.flush(timeout=5)
                self.producer.close(timeout=5)
                logger.info("Kafka producer closed")
            except Exception:
                pass
