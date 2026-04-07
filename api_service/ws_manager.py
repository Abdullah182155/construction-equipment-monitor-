"""
WebSocket Connection Manager.
Handles real-time broadcast of annotated frames and equipment events.
"""

import logging
import asyncio
import json
import base64
from typing import Set, Dict, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time data streaming.
    
    Supports two types of messages:
    1. Binary: Base64-encoded JPEG frames for video display
    2. JSON: Equipment state/event updates for dashboard
    """

    def __init__(self):
        self.active_connections: Set = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket):
        """Accept and store a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket):
        """Remove a disconnected WebSocket."""
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast_frame(self, frame: np.ndarray, quality: int = 60):
        """
        Broadcast an annotated video frame to all connected clients.
        
        Encodes frame as JPEG, then base64, and sends as JSON message
        with type="frame".
        """
        if not self.active_connections:
            return

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        frame_b64 = base64.b64encode(buffer).decode("utf-8")

        message = json.dumps({
            "type": "frame",
            "data": frame_b64
        })

        await self._broadcast(message)

    async def broadcast_event(self, event_data: Dict[str, Any]):
        """
        Broadcast an equipment event update to all connected clients.
        """
        message = json.dumps({
            "type": "event",
            "data": event_data
        })
        await self._broadcast(message)

    async def broadcast_summary(self, summary: Dict[str, Any]):
        """Broadcast aggregate summary to all clients."""
        message = json.dumps({
            "type": "summary",
            "data": summary
        })
        await self._broadcast(message)

    async def _broadcast(self, message: str):
        """Send message to all active connections, removing dead ones."""
        dead = set()
        async with self._lock:
            connections = self.active_connections.copy()

        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)

        if dead:
            async with self._lock:
                self.active_connections -= dead

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)
