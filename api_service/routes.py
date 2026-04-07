"""
REST API Routes for Equipment Monitoring.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["equipment"])

# These will be set by main.py during app startup
_repository = None


def set_repository(repo):
    """Set the database repository (called from main.py)."""
    global _repository
    _repository = repo


@router.get("/status")
async def system_status():
    """System health check endpoint."""
    return {
        "status": "online",
        "service": "Equipment Monitoring API",
        "version": "1.0.0"
    }


@router.get("/equipment")
async def list_equipment():
    """List all tracked equipment with latest state."""
    if _repository is None:
        raise HTTPException(503, "Database not available")
    try:
        equipment = _repository.get_all_equipment()
        return {"equipment": equipment, "count": len(equipment)}
    except Exception as e:
        logger.error(f"Failed to fetch equipment: {e}")
        raise HTTPException(500, str(e))


@router.get("/equipment/{equipment_id}/history")
async def equipment_history(
    equipment_id: int,
    limit: int = Query(100, ge=1, le=1000),
    since_minutes: int = Query(60, ge=1, le=1440)
):
    """Get event history for a specific piece of equipment."""
    if _repository is None:
        raise HTTPException(503, "Database not available")
    try:
        history = _repository.get_equipment_history(
            equipment_id, limit=limit, since_minutes=since_minutes
        )
        return {"equipment_id": equipment_id, "events": history, "count": len(history)}
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        raise HTTPException(500, str(e))


@router.get("/metrics")
async def aggregate_metrics():
    """Get aggregate utilization metrics across all equipment."""
    if _repository is None:
        raise HTTPException(503, "Database not available")
    try:
        metrics = _repository.get_aggregate_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to fetch metrics: {e}")
        raise HTTPException(500, str(e))


@router.get("/activities")
async def activity_distribution(equipment_id: Optional[int] = None):
    """Get activity distribution (count per activity type)."""
    if _repository is None:
        raise HTTPException(503, "Database not available")
    try:
        dist = _repository.get_activity_distribution(equipment_id)
        return {"distribution": dist}
    except Exception as e:
        logger.error(f"Failed to fetch activity distribution: {e}")
        raise HTTPException(500, str(e))
