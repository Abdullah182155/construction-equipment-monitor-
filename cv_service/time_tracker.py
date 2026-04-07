"""
Time Tracker for Per-Equipment Utilization Metrics.

Tracks active time, idle time, and computes utilization rate
for each piece of equipment identified by tracker ID.
"""

import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EquipmentStats:
    """Statistics for a single piece of equipment."""
    tracker_id: int
    first_seen: float = 0.0
    last_seen: float = 0.0
    active_time: float = 0.0
    idle_time: float = 0.0
    last_state: str = "INACTIVE"
    last_activity: str = "Waiting"
    last_state_change: float = 0.0
    frame_count: int = 0

    @property
    def total_time(self) -> float:
        return self.active_time + self.idle_time

    @property
    def utilization(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.active_time / self.total_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equipment_id": self.tracker_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "active_time": round(self.active_time, 2),
            "idle_time": round(self.idle_time, 2),
            "total_time": round(self.total_time, 2),
            "utilization": round(self.utilization, 4),
            "state": self.last_state,
            "activity": self.last_activity,
            "frame_count": self.frame_count
        }


class TimeTracker:
    """
    Tracks utilization metrics for all detected equipment.
    
    Updates are called per-frame with the current state of each
    tracked object. Time deltas are computed from actual timestamps.
    """

    def __init__(self):
        self._equipment: Dict[int, EquipmentStats] = {}
        self._fps: float = 30.0

    def set_fps(self, fps: float):
        """Set video FPS for time calculations."""
        self._fps = max(fps, 1.0)

    def update(
        self,
        tracker_id: int,
        is_active: bool,
        activity: str,
        timestamp: Optional[float] = None
    ) -> EquipmentStats:
        """
        Update time tracking for a specific equipment.

        Args:
            tracker_id: Persistent equipment ID
            is_active: Current active state
            activity: Current activity classification
            timestamp: Current time (defaults to time.time())

        Returns:
            Updated EquipmentStats
        """
        now = timestamp or time.time()
        state = "ACTIVE" if is_active else "INACTIVE"

        if tracker_id not in self._equipment:
            self._equipment[tracker_id] = EquipmentStats(
                tracker_id=tracker_id,
                first_seen=now,
                last_seen=now,
                last_state_change=now
            )
            logger.info(f"New equipment tracked: ID={tracker_id}")

        stats = self._equipment[tracker_id]
        dt = now - stats.last_seen if stats.last_seen > 0 else 1.0 / self._fps

        # Clamp dt to avoid huge jumps (e.g., video seeking)
        dt = min(dt, 2.0)

        # Accumulate time based on previous state
        if stats.last_state == "ACTIVE":
            stats.active_time += dt
        else:
            stats.idle_time += dt

        # Update state
        stats.last_seen = now
        stats.last_state = state
        stats.last_activity = activity
        stats.frame_count += 1

        if state != stats.last_state:
            stats.last_state_change = now

        return stats

    def get_stats(self, tracker_id: int) -> Optional[EquipmentStats]:
        """Get stats for a specific equipment."""
        return self._equipment.get(tracker_id)

    def get_all_stats(self) -> Dict[int, EquipmentStats]:
        """Get stats for all tracked equipment."""
        return self._equipment.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregate summary across all equipment."""
        if not self._equipment:
            return {
                "total_equipment": 0,
                "active_count": 0,
                "inactive_count": 0,
                "avg_utilization": 0.0
            }

        stats_list = list(self._equipment.values())
        active_count = sum(1 for s in stats_list if s.last_state == "ACTIVE")
        utilizations = [s.utilization for s in stats_list]

        return {
            "total_equipment": len(stats_list),
            "active_count": active_count,
            "inactive_count": len(stats_list) - active_count,
            "avg_utilization": round(sum(utilizations) / len(utilizations), 4) if utilizations else 0.0,
            "equipment": {s.tracker_id: s.to_dict() for s in stats_list}
        }

    def reset(self):
        """Reset all tracking data."""
        self._equipment.clear()
