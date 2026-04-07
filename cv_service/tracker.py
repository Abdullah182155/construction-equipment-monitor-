"""
Centroid-Distance Object Tracker for Construction Equipment.
Assigns persistent IDs based on bounding box center distance.

Key features for construction equipment:
  - Distance threshold scales with bbox size (big equipment = bigger tolerance)
  - Same-class matching priority (excavator matches excavator, not truck)
  - Motion metrics for ACTIVE/IDLE state detection
"""

import logging
from typing import List, Dict, Any, Tuple
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class EquipmentTracker:
    """
    Tracks equipment using centroid distance + class matching.
    Distance tolerance scales with bounding box diagonal so
    large equipment (excavators) get a generous matching radius.
    """

    def __init__(self, max_lost: int = 120):
        """
        Args:
            max_lost: Frames a track survives without detection before removal.
        """
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.motion_window = 10
        self.motion_threshold = 5.0  # px/frame
        # Distance threshold = this fraction of the bbox diagonal
        self.distance_ratio = 1.0
        logger.info(f"Centroid tracker initialized (max_lost={max_lost})")

    @staticmethod
    def _center(bbox: List[int]) -> Tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _diagonal(bbox: List[int]) -> float:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return float(np.sqrt(w**2 + h**2))

    @staticmethod
    def _bbox_area(bbox: List[int]) -> float:
        return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """Update tracker and return tracked objects with motion info."""
        # Age all tracks
        for tid in list(self.tracks):
            self.tracks[tid]["lost"] += 1

        if not detections:
            self._purge_lost()
            return []

        det_centers = [self._center(d["bbox"]) for d in detections]
        det_diags = [self._diagonal(d["bbox"]) for d in detections]

        if not self.tracks:
            results = []
            for i, det in enumerate(detections):
                tid = self._register(det, det_centers[i])
                results.append(self._make_result(tid, det))
            return results

        # Build cost matrix with class mismatch penalty
        track_ids = list(self.tracks.keys())
        track_centers = [self.tracks[tid]["center"] for tid in track_ids]
        track_classes = [self.tracks[tid]["class_id"] for tid in track_ids]
        track_diags = [self._diagonal(self.tracks[tid]["bbox"]) for tid in track_ids]

        cost_matrix = np.full((len(track_ids), len(detections)), 1e6, dtype=np.float64)

        for t in range(len(track_ids)):
            for d in range(len(detections)):
                tc = track_centers[t]
                dc = det_centers[d]
                dist = np.sqrt((tc[0] - dc[0])**2 + (tc[1] - dc[1])**2)

                # Max allowed distance = fraction of the larger bbox diagonal
                max_dist = self.distance_ratio * max(track_diags[t], det_diags[d])
                max_dist = max(max_dist, 100.0)  # Minimum 100px

                # Different class? Add heavy penalty (but don't block completely)
                class_penalty = 0.0
                if track_classes[t] != detections[d]["class_id"]:
                    class_penalty = max_dist * 2.0  # Makes cross-class match very unlikely

                if dist <= max_dist:
                    cost_matrix[t, d] = dist + class_penalty
                # Else stays at 1e6 (impossible match)

        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_dets = set()
        results = []

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] < 1e5:  # Valid match
                tid = track_ids[r]
                det = detections[c]
                self._update_track(tid, det, det_centers[c])
                matched_dets.add(c)
                results.append(self._make_result(tid, det))

        # Unmatched detections → new tracks
        for d in range(len(detections)):
            if d not in matched_dets:
                tid = self._register(detections[d], det_centers[d])
                results.append(self._make_result(tid, detections[d]))

        self._purge_lost()
        logger.debug(f"Tracking {len(results)} objects, {len(self.tracks)} active tracks")
        return results

    def _register(self, detection: Dict[str, Any], center: Tuple[float, float]) -> int:
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = {
            "center": center,
            "bbox": detection["bbox"],
            "lost": 0,
            "confidence": detection["confidence"],
            "class_name": detection["class_name"],
            "class_id": detection["class_id"],
            "center_history": deque([center], maxlen=self.motion_window),
            "area_history": deque([self._bbox_area(detection["bbox"])], maxlen=self.motion_window),
        }
        return tid

    def _update_track(self, tid: int, detection: Dict[str, Any], center: Tuple[float, float]):
        track = self.tracks[tid]
        track["center"] = center
        track["bbox"] = detection["bbox"]
        track["lost"] = 0
        track["confidence"] = detection["confidence"]
        track["class_name"] = detection["class_name"]
        track["class_id"] = detection["class_id"]
        track["center_history"].append(center)
        track["area_history"].append(self._bbox_area(detection["bbox"]))

    def get_motion_info(self, tracker_id: int) -> Dict[str, Any]:
        """Compute motion metrics for a tracked object."""
        if tracker_id not in self.tracks:
            return {"velocity": 0.0, "bbox_change": 0.0, "is_moving": False}

        track = self.tracks[tracker_id]
        history = list(track["center_history"])
        areas = list(track["area_history"])

        if len(history) < 2:
            return {"velocity": 0.0, "bbox_change": 0.0, "is_moving": False}

        # Average velocity
        velocities = []
        for i in range(1, len(history)):
            dx = history[i][0] - history[i-1][0]
            dy = history[i][1] - history[i-1][1]
            velocities.append(np.sqrt(dx**2 + dy**2))
        avg_velocity = float(np.mean(velocities))

        # Bbox area change (excavator arm movement)
        if len(areas) >= 2 and areas[0] > 0:
            area_changes = []
            for i in range(1, len(areas)):
                if areas[i-1] > 0:
                    area_changes.append(abs(areas[i] - areas[i-1]) / areas[i-1])
            avg_bbox_change = float(np.mean(area_changes)) if area_changes else 0.0
        else:
            avg_bbox_change = 0.0

        is_moving = avg_velocity > self.motion_threshold or avg_bbox_change > 0.08

        return {
            "velocity": round(avg_velocity, 2),
            "bbox_change": round(avg_bbox_change, 4),
            "is_moving": is_moving
        }

    def _make_result(self, tid: int, detection: Dict[str, Any]) -> Dict[str, Any]:
        motion = self.get_motion_info(tid)
        return {
            "tracker_id": tid,
            "bbox": detection["bbox"],
            "confidence": detection["confidence"],
            "class_id": detection["class_id"],
            "class_name": detection["class_name"],
            "velocity": motion["velocity"],
            "bbox_change": motion["bbox_change"],
            "is_moving": motion["is_moving"],
        }

    def _purge_lost(self):
        dead = [tid for tid, t in self.tracks.items() if t["lost"] > self.max_lost]
        for tid in dead:
            del self.tracks[tid]

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
