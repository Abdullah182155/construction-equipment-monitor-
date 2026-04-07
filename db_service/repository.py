"""
Repository Pattern for Database Operations.
CRUD operations for equipment events and summaries.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from db_service.models import EquipmentEvent, EquipmentSummary

logger = logging.getLogger(__name__)


class EquipmentRepository:
    """
    Data access layer for equipment monitoring data.
    Handles inserts, queries, and aggregations.
    """

    def __init__(self, database):
        self.db = database

    def insert_event(self, event_data: Dict[str, Any]):
        """
        Insert a single equipment event and update summary.
        
        Args:
            event_data: Dict from Kafka message
        """
        session = self.db.get_session()
        try:
            # Parse timestamp
            ts = event_data.get("timestamp")
            if isinstance(ts, str):
                try:
                    timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    timestamp = datetime.utcnow()
            else:
                timestamp = datetime.utcnow()

            # Insert event
            event = EquipmentEvent(
                equipment_id=event_data.get("equipment_id", 0),
                timestamp=timestamp,
                state=event_data.get("state", "INACTIVE"),
                activity=event_data.get("activity", "Waiting"),
                active_time=event_data.get("active_time", 0),
                idle_time=event_data.get("idle_time", 0),
                utilization=event_data.get("utilization", 0),
                confidence=event_data.get("confidence", 0),
                algorithm=event_data.get("algorithm", "rule_based"),
                motion_score=event_data.get("motion_score", 0),
                bbox=event_data.get("bbox"),
            )
            session.add(event)

            # Update or create summary
            eq_id = event_data.get("equipment_id", 0)
            summary = session.query(EquipmentSummary).filter_by(
                equipment_id=eq_id
            ).first()

            if summary:
                summary.last_seen = timestamp
                summary.state = event_data.get("state", "INACTIVE")
                summary.activity = event_data.get("activity", "Waiting")
                summary.total_active_time = event_data.get("active_time", 0)
                summary.total_idle_time = event_data.get("idle_time", 0)
                summary.utilization = event_data.get("utilization", 0)
                summary.event_count += 1
                summary.last_confidence = event_data.get("confidence", 0)
                summary.last_algorithm = event_data.get("algorithm", "rule_based")
            else:
                summary = EquipmentSummary(
                    equipment_id=eq_id,
                    last_seen=timestamp,
                    state=event_data.get("state", "INACTIVE"),
                    activity=event_data.get("activity", "Waiting"),
                    total_active_time=event_data.get("active_time", 0),
                    total_idle_time=event_data.get("idle_time", 0),
                    utilization=event_data.get("utilization", 0),
                    event_count=1,
                    last_confidence=event_data.get("confidence", 0),
                    last_algorithm=event_data.get("algorithm", "rule_based"),
                )
                session.add(summary)

            session.commit()
            logger.debug(f"Event stored for equipment #{eq_id}")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to insert event: {e}")
            raise
        finally:
            session.close()

    def get_all_equipment(self) -> List[Dict[str, Any]]:
        """Get latest state for all tracked equipment."""
        session = self.db.get_session()
        try:
            summaries = session.query(EquipmentSummary).order_by(
                EquipmentSummary.equipment_id
            ).all()
            return [s.to_dict() for s in summaries]
        finally:
            session.close()

    def get_equipment_history(
        self,
        equipment_id: int,
        limit: int = 100,
        since_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """Get event history for a specific equipment."""
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
            events = session.query(EquipmentEvent).filter(
                EquipmentEvent.equipment_id == equipment_id,
                EquipmentEvent.timestamp >= cutoff
            ).order_by(desc(EquipmentEvent.timestamp)).limit(limit).all()
            return [e.to_dict() for e in events]
        finally:
            session.close()

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics across all equipment."""
        session = self.db.get_session()
        try:
            summaries = session.query(EquipmentSummary).all()
            if not summaries:
                return {
                    "total_equipment": 0,
                    "active_count": 0,
                    "inactive_count": 0,
                    "avg_utilization": 0.0,
                    "total_events": 0
                }

            active = sum(1 for s in summaries if s.state == "ACTIVE")
            utils = [s.utilization for s in summaries]
            total_events = sum(s.event_count for s in summaries)

            return {
                "total_equipment": len(summaries),
                "active_count": active,
                "inactive_count": len(summaries) - active,
                "avg_utilization": round(sum(utils) / len(utils), 4) if utils else 0.0,
                "total_events": total_events,
                "equipment": [s.to_dict() for s in summaries]
            }
        finally:
            session.close()

    def get_activity_distribution(self, equipment_id: Optional[int] = None) -> Dict[str, int]:
        """Get activity distribution counts."""
        session = self.db.get_session()
        try:
            query = session.query(
                EquipmentEvent.activity,
                func.count(EquipmentEvent.id).label("count")
            )
            if equipment_id is not None:
                query = query.filter(EquipmentEvent.equipment_id == equipment_id)
            query = query.group_by(EquipmentEvent.activity)

            return {row.activity: row.count for row in query.all()}
        finally:
            session.close()
