"""
Motion Analysis Module with Strategy Pattern.
Implements three algorithms for motion detection within tracked bounding boxes:
1. Optical Flow (Farneback) — recommended for real-time state detection
2. Frame Differencing — lightweight fallback
Both use region-based analysis (3x3 grid) to handle articulated parts.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class MotionResult:
    """Container for motion analysis results."""

    __slots__ = [
        "is_active", "motion_score", "region_scores",
        "flow_directions", "active_regions"
    ]

    def __init__(
        self,
        is_active: bool = False,
        motion_score: float = 0.0,
        region_scores: Optional[np.ndarray] = None,
        flow_directions: Optional[np.ndarray] = None,
        active_regions: Optional[List[Tuple[int, int]]] = None
    ):
        self.is_active = is_active
        self.motion_score = motion_score
        self.region_scores = region_scores if region_scores is not None else np.zeros((3, 3))
        self.flow_directions = flow_directions if flow_directions is not None else np.zeros((3, 3, 2))
        self.active_regions = active_regions or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_active": self.is_active,
            "motion_score": round(self.motion_score, 4),
            "region_scores": self.region_scores.tolist(),
            "active_regions": self.active_regions,
        }


class BaseMotionAnalyzer(ABC):
    """Abstract base class for motion analyzers (Strategy Pattern)."""

    @abstractmethod
    def analyze(
        self, 
        current_frame: np.ndarray,
        previous_frame: np.ndarray,
        bbox: List[int],
        tracker_id: int
    ) -> MotionResult:
        """Analyze motion within a bounding box between two frames."""
        pass

    def _extract_roi(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Extract region of interest from frame, with bounds checking."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        return frame[y1:y2, x1:x2]

    def _to_gray(self, roi: np.ndarray) -> np.ndarray:
        """Convert ROI to grayscale."""
        if len(roi.shape) == 3:
            return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return roi


class OpticalFlowAnalyzer(BaseMotionAnalyzer):
    """
    Dense Optical Flow (Farneback) motion analyzer with camera motion compensation.
    
    Splits bounding box into a 3x3 grid for region-based analysis.
    Subtracts global (camera) motion so only real equipment movement is detected.
    
    Recommended for: Real-time ACTIVE/INACTIVE state detection.
    """

    def __init__(self, config):
        self.config = config
        self.grid_rows = config.grid_rows
        self.grid_cols = config.grid_cols
        self.threshold = config.threshold
        self.active_zone_threshold = config.active_zone_threshold
        # Farneback parameters — tuned for speed
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=2,
            winsize=9,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0
        )
        # Cache for global flow (computed once per frame pair)
        self._cached_global_flow = None
        self._cached_frame_id = None

    def compute_global_flow(self, current_frame: np.ndarray, previous_frame: np.ndarray):
        """
        Compute global (camera) motion as median optical flow over the full frame.
        This is cached per frame pair so it's only computed once.
        """
        frame_id = id(current_frame)
        if self._cached_frame_id == frame_id:
            return self._cached_global_flow

        # Downscale full frame for fast global flow computation
        scale = 160.0 / max(current_frame.shape[:2])
        if scale < 1.0:
            small_curr = cv2.resize(current_frame, None, fx=scale, fy=scale)
            small_prev = cv2.resize(previous_frame, None, fx=scale, fy=scale)
        else:
            small_curr = current_frame
            small_prev = previous_frame

        gray_curr = self._to_gray(small_curr)
        gray_prev = self._to_gray(small_prev)

        if gray_curr.shape != gray_prev.shape:
            gray_prev = cv2.resize(gray_prev, (gray_curr.shape[1], gray_curr.shape[0]))

        try:
            flow = cv2.calcOpticalFlowFarneback(
                gray_prev, gray_curr, None, **self.flow_params
            )
            # Median flow = camera motion estimate
            global_dx = float(np.median(flow[..., 0]))
            global_dy = float(np.median(flow[..., 1]))
        except cv2.error:
            global_dx, global_dy = 0.0, 0.0

        self._cached_global_flow = (global_dx, global_dy)
        self._cached_frame_id = frame_id
        return self._cached_global_flow

    def analyze(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray,
        bbox: List[int],
        tracker_id: int
    ) -> MotionResult:
        """
        Compute dense optical flow within bounding box regions,
        with camera motion subtracted.
        """
        roi_curr = self._extract_roi(current_frame, bbox)
        roi_prev = self._extract_roi(previous_frame, bbox)

        if roi_curr is None or roi_prev is None:
            return MotionResult()

        gray_curr = self._to_gray(roi_curr)
        gray_prev = self._to_gray(roi_prev)

        # Resize to match if shapes differ
        if gray_curr.shape != gray_prev.shape:
            gray_prev = cv2.resize(gray_prev, (gray_curr.shape[1], gray_curr.shape[0]))

        # Compute dense optical flow within bounding box
        try:
            flow = cv2.calcOpticalFlowFarneback(
                gray_prev, gray_curr, None, **self.flow_params
            )
        except cv2.error:
            return MotionResult()

        # Subtract camera motion (global flow)
        global_dx, global_dy = self.compute_global_flow(current_frame, previous_frame)
        flow[..., 0] -= global_dx
        flow[..., 1] -= global_dy

        # Region-based analysis
        h, w = gray_curr.shape
        region_scores = np.zeros((self.grid_rows, self.grid_cols))
        flow_directions = np.zeros((self.grid_rows, self.grid_cols, 2))
        active_regions = []

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                r_start = r * h // self.grid_rows
                r_end = (r + 1) * h // self.grid_rows
                c_start = c * w // self.grid_cols
                c_end = (c + 1) * w // self.grid_cols

                region_flow = flow[r_start:r_end, c_start:c_end]
                if region_flow.size == 0:
                    continue

                # Compute magnitude
                mag, ang = cv2.cartToPolar(region_flow[..., 0], region_flow[..., 1])
                region_scores[r, c] = float(np.mean(mag))
                flow_directions[r, c] = [
                    float(np.mean(region_flow[..., 0])),
                    float(np.mean(region_flow[..., 1]))
                ]

                if region_scores[r, c] > self.threshold:
                    active_regions.append((r, c))

        overall_score = float(np.mean(region_scores))
        is_active = len(active_regions) >= self.active_zone_threshold

        return MotionResult(
            is_active=is_active,
            motion_score=overall_score,
            region_scores=region_scores,
            flow_directions=flow_directions,
            active_regions=active_regions
        )


class FrameDiffAnalyzer(BaseMotionAnalyzer):
    """
    Frame differencing motion analyzer (lightweight fallback).
    
    Uses absolute pixel difference between consecutive frames
    within bounding box regions.
    """

    def __init__(self, config):
        self.config = config
        self.grid_rows = config.grid_rows
        self.grid_cols = config.grid_cols
        self.threshold = config.threshold * 10  # Scale for pixel diff
        self.active_zone_threshold = config.active_zone_threshold

    def analyze(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray,
        bbox: List[int],
        tracker_id: int
    ) -> MotionResult:
        roi_curr = self._extract_roi(current_frame, bbox)
        roi_prev = self._extract_roi(previous_frame, bbox)

        if roi_curr is None or roi_prev is None:
            return MotionResult()

        gray_curr = self._to_gray(roi_curr)
        gray_prev = self._to_gray(roi_prev)

        if gray_curr.shape != gray_prev.shape:
            gray_prev = cv2.resize(gray_prev, (gray_curr.shape[1], gray_curr.shape[0]))

        diff = cv2.absdiff(gray_curr, gray_prev).astype(np.float32)

        h, w = gray_curr.shape
        region_scores = np.zeros((self.grid_rows, self.grid_cols))
        active_regions = []

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                r_start = r * h // self.grid_rows
                r_end = (r + 1) * h // self.grid_rows
                c_start = c * w // self.grid_cols
                c_end = (c + 1) * w // self.grid_cols

                region_diff = diff[r_start:r_end, c_start:c_end]
                if region_diff.size == 0:
                    continue

                region_scores[r, c] = float(np.mean(region_diff))
                if region_scores[r, c] > self.threshold:
                    active_regions.append((r, c))

        overall_score = float(np.mean(region_scores))
        is_active = len(active_regions) >= self.active_zone_threshold

        return MotionResult(
            is_active=is_active,
            motion_score=overall_score,
            region_scores=region_scores,
            active_regions=active_regions
        )


class MotionAnalyzerFactory:
    """Factory to create the appropriate motion analyzer based on config."""

    @staticmethod
    def create(config) -> BaseMotionAnalyzer:
        algorithm = config.algorithm
        if algorithm == "optical_flow":
            logger.info("Using Optical Flow motion analyzer")
            return OpticalFlowAnalyzer(config)
        elif algorithm == "frame_diff":
            logger.info("Using Frame Differencing motion analyzer")
            return FrameDiffAnalyzer(config)
        else:
            logger.warning(f"Unknown algorithm '{algorithm}', defaulting to optical_flow")
            return OpticalFlowAnalyzer(config)


class TemporalSmoother:
    """
    Temporal smoothing for motion state decisions.
    Uses a sliding window of recent states to reduce flickering.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._histories: Dict[int, deque] = {}

    def smooth(self, tracker_id: int, is_active: bool) -> bool:
        """
        Apply temporal smoothing to the active state.
        Returns smoothed state based on majority vote.
        """
        if tracker_id not in self._histories:
            self._histories[tracker_id] = deque(maxlen=self.window_size)

        self._histories[tracker_id].append(is_active)
        history = self._histories[tracker_id]

        # Majority vote
        active_count = sum(history)
        return active_count > len(history) / 2

    def clear(self, tracker_id: int):
        """Clear history for a specific tracker ID."""
        self._histories.pop(tracker_id, None)
