"""
Feature Extractor for LSTM and C3D models.
Extracts per-frame feature vectors from tracked objects for temporal modeling.
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts feature vectors from tracked objects for LSTM/C3D input.
    
    Feature vector per frame (27 dimensions):
        - Optical flow region magnitudes: 9 values (3x3 grid)
        - Optical flow region directions (x component): 9 values
        - Bounding box kinematics: 9 values
            - normalized x_center, y_center
            - normalized width, height
            - aspect_ratio
            - dx (velocity x), dy (velocity y)
            - dw (width change), dh (height change)
    """

    def __init__(self, config, frame_shape: Tuple[int, int] = (1080, 1920)):
        self.config = config
        self.sequence_length = config.sequence_length
        self.frame_h, self.frame_w = frame_shape
        # Per-tracker feature buffers
        self._buffers: Dict[int, deque] = {}
        # Previous bbox for kinematics
        self._prev_bboxes: Dict[int, List[int]] = {}

    def update_frame_shape(self, h: int, w: int):
        """Update frame dimensions for normalization."""
        self.frame_h = h
        self.frame_w = w

    def extract(
        self,
        tracker_id: int,
        bbox: List[int],
        region_scores: np.ndarray,
        flow_directions: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract and buffer feature vector for a tracked object.

        Args:
            tracker_id: Persistent object ID
            bbox: [x1, y1, x2, y2]
            region_scores: 3x3 motion magnitude scores
            flow_directions: 3x3x2 flow direction vectors

        Returns:
            Feature sequence (seq_len, 27) if buffer is full, else None
        """
        # Compute bbox kinematics
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0 / max(self.frame_w, 1)
        cy = (y1 + y2) / 2.0 / max(self.frame_h, 1)
        w = (x2 - x1) / max(self.frame_w, 1)
        h = (y2 - y1) / max(self.frame_h, 1)
        aspect = w / max(h, 1e-6)

        # Velocity (change from previous frame)
        dx, dy, dw, dh = 0.0, 0.0, 0.0, 0.0
        if tracker_id in self._prev_bboxes:
            prev = self._prev_bboxes[tracker_id]
            px1, py1, px2, py2 = prev
            pcx = (px1 + px2) / 2.0 / max(self.frame_w, 1)
            pcy = (py1 + py2) / 2.0 / max(self.frame_h, 1)
            pw = (px2 - px1) / max(self.frame_w, 1)
            ph = (py2 - py1) / max(self.frame_h, 1)
            dx = cx - pcx
            dy = cy - pcy
            dw = w - pw
            dh = h - ph

        self._prev_bboxes[tracker_id] = bbox

        # Build feature vector
        flow_mags = region_scores.flatten()  # 9 values
        flow_dirs = flow_directions[..., 0].flatten()  # 9 values (x-component)
        kinematics = np.array([cx, cy, w, h, aspect, dx, dy, dw, dh])

        feature = np.concatenate([flow_mags, flow_dirs, kinematics]).astype(np.float32)

        # Buffer management
        if tracker_id not in self._buffers:
            self._buffers[tracker_id] = deque(maxlen=self.sequence_length)

        self._buffers[tracker_id].append(feature)

        # Return sequence only if buffer is full
        if len(self._buffers[tracker_id]) >= self.sequence_length:
            return np.array(list(self._buffers[tracker_id]))

        return None

    def get_sequence(self, tracker_id: int) -> Optional[np.ndarray]:
        """Get current feature sequence for a tracker (if available)."""
        if tracker_id in self._buffers and len(self._buffers[tracker_id]) >= self.sequence_length:
            return np.array(list(self._buffers[tracker_id]))
        return None

    def clear(self, tracker_id: int):
        """Clear buffers for a tracker."""
        self._buffers.pop(tracker_id, None)
        self._prev_bboxes.pop(tracker_id, None)


class ClipBuffer:
    """
    Frame clip buffer for C3D model input.
    Stores cropped & resized frame clips from bounding boxes.
    """

    def __init__(self, clip_length: int = 16, crop_size: int = 112):
        self.clip_length = clip_length
        self.crop_size = crop_size
        self._buffers: Dict[int, deque] = {}

    def add_frame(
        self, tracker_id: int, frame: np.ndarray, bbox: List[int]
    ) -> Optional[np.ndarray]:
        """
        Add a cropped frame to the clip buffer.

        Returns:
            Clip array (clip_length, 3, crop_size, crop_size) if buffer full, else None
        """
        import cv2

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.crop_size, self.crop_size))
        # Convert to CHW format and normalize
        crop = crop.astype(np.float32) / 255.0
        crop = np.transpose(crop, (2, 0, 1))  # HWC -> CHW

        if tracker_id not in self._buffers:
            self._buffers[tracker_id] = deque(maxlen=self.clip_length)

        self._buffers[tracker_id].append(crop)

        if len(self._buffers[tracker_id]) >= self.clip_length:
            return np.array(list(self._buffers[tracker_id]))

        return None

    def clear(self, tracker_id: int):
        """Clear clip buffer for a tracker."""
        self._buffers.pop(tracker_id, None)
