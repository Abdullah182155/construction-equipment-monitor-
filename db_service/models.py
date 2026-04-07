"""
SQLAlchemy ORM Models for Equipment Monitoring.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, JSON, Index, Text
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class EquipmentEvent(Base):
    """
    Individual equipment monitoring events.
    Each row represents a single observation of a piece of equipment.
    """
    __tablename__ = "equipment_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    equipment_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    state = Column(String(20), nullable=False)  # ACTIVE / INACTIVE
    activity = Column(String(50), nullable=False)  # Digging, Swinging/Loading, etc.
    active_time = Column(Float, default=0.0)
    idle_time = Column(Float, default=0.0)
    utilization = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0)
    algorithm = Column(String(20), default="rule_based")
    motion_score = Column(Float, default=0.0)
    bbox = Column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_equipment_events_eq_ts", "equipment_id", "timestamp"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "equipment_id": self.equipment_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "state": self.state,
            "activity": self.activity,
            "active_time": self.active_time,
            "idle_time": self.idle_time,
            "utilization": self.utilization,
            "confidence": self.confidence,
            "algorithm": self.algorithm,
            "motion_score": self.motion_score,
            "bbox": self.bbox,
        }


class EquipmentSummary(Base):
    """
    Latest aggregated stats per equipment.
    Updated on each new event for fast dashboard queries.
    """
    __tablename__ = "equipment_summary"

    equipment_id = Column(Integer, primary_key=True)
    last_seen = Column(DateTime, nullable=False)
    state = Column(String(20), nullable=False)
    activity = Column(String(50), nullable=False)
    total_active_time = Column(Float, default=0.0)
    total_idle_time = Column(Float, default=0.0)
    utilization = Column(Float, default=0.0)
    event_count = Column(Integer, default=0)
    last_confidence = Column(Float, default=0.0)
    last_algorithm = Column(String(20), default="rule_based")

    def to_dict(self):
        return {
            "equipment_id": self.equipment_id,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "state": self.state,
            "activity": self.activity,
            "total_active_time": self.total_active_time,
            "total_idle_time": self.total_idle_time,
            "utilization": self.utilization,
            "event_count": self.event_count,
            "last_confidence": self.last_confidence,
            "last_algorithm": self.last_algorithm,
        }
