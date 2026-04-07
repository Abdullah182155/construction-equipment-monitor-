"""
CV Pipeline Orchestrator.

Supports two modes:
  - Custom mode (best.pt): Detects equipment TYPES (excavator, dump_truck, etc.)
    Activity state determined by tracker motion metrics (no optical flow).
  - Generic mode (yolov8n.pt): COCO classes + Optical Flow + Classifier.

Pipeline: YOLO → Centroid Track (with motion) → Time Tracking → Annotated Frame
"""

import logging
import time
from typing import Dict, Any, Generator, Optional, List
import numpy as np
import cv2

from config.settings import settings
from cv_service.detector import EquipmentDetector
from cv_service.tracker import EquipmentTracker
from cv_service.time_tracker import TimeTracker

logger = logging.getLogger(__name__)

# Equipment-type specific colors for bounding boxes
EQUIPMENT_COLORS = {
    "Excavator":      (0, 220, 100),    # Green
    "Dump Truck":     (0, 180, 255),    # Orange
    "Concrete Mixer": (255, 180, 0),    # Blue
}

# State colors
STATE_COLORS = {
    "ACTIVE":  (0, 220, 100),   # Green
    "IDLE":    (0, 100, 230),   # Red-ish
}


class CVPipeline:
    """
    Main CV pipeline for construction equipment monitoring.
    
    In custom mode, YOLO detects equipment type and the centroid tracker
    determines ACTIVE/IDLE state from position + bbox changes.
    """

    def __init__(self):
        logger.info("Initializing CV Pipeline...")

        self.pipeline_mode = settings.pipeline_mode

        # Core modules
        self.detector = EquipmentDetector(settings.yolo)
        self.tracker = EquipmentTracker()
        self.time_tracker = TimeTracker()

        # Generic mode modules (lazy loaded)
        self.motion_analyzer = None
        self.temporal_smoother = None
        self.activity_classifier = None

        # We always initialize motion/activity modules now, so Excavators can get
        # detailed activity classification (Digging, etc.) using optical flow
        self._init_generic_modules()

        # State
        self.previous_frame = None
        self.frame_count = 0
        self.fps = 30.0

        # Performance
        self.frame_skip = settings.video.frame_skip
        self.process_width = settings.video.process_width
        self._last_result = None

        logger.info(f"CV Pipeline initialized — mode={self.pipeline_mode}")
        logger.info(f"  frame_skip={self.frame_skip}, process_width={self.process_width}")

    def _init_generic_modules(self):
        """Load optical flow + classifiers for generic COCO model."""
        from cv_service.motion_analyzer import MotionAnalyzerFactory, TemporalSmoother
        from cv_service.feature_extractor import FeatureExtractor, ClipBuffer
        from cv_service.activity_classifier import HybridActivityClassifier

        self.motion_analyzer = MotionAnalyzerFactory.create(settings.motion)
        self.temporal_smoother = TemporalSmoother(settings.motion.temporal_smoothing_frames)
        self.feature_extractor = FeatureExtractor(settings.lstm)
        self.clip_buffer = ClipBuffer(
            clip_length=settings.c3d.clip_length,
            crop_size=settings.c3d.crop_size
        )

        lstm_model = None
        c3d_model = None
        algo = settings.motion.activity_algorithm
        if algo in ("lstm", "hybrid"):
            try:
                from cv_service.lstm_model import LSTMActivityModel
                lstm_model = LSTMActivityModel(settings.lstm)
            except Exception as e:
                logger.warning(f"LSTM not available: {e}")
        if algo in ("c3d", "hybrid"):
            try:
                from cv_service.c3d_model import C3DModel
                c3d_model = C3DModel(settings.c3d)
            except Exception as e:
                logger.warning(f"C3D not available: {e}")

        self.activity_classifier = HybridActivityClassifier(
            config=settings.motion,
            lstm_model=lstm_model,
            c3d_model=c3d_model
        )
        logger.info("Generic pipeline modules loaded")

    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self.process_width:
            return frame
        scale = self.process_width / w
        return cv2.resize(frame, (self.process_width, int(h * scale)), interpolation=cv2.INTER_LINEAR)

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Process a single video frame through the pipeline."""
        now = timestamp or time.time()
        self.frame_count += 1

        proc_frame = self._downscale(frame)

        # 1. Detect equipment
        detections = self.detector.detect(proc_frame)

        # 2. Track objects (returns motion info too)
        tracked_objects = self.tracker.update(detections, proc_frame)

        # 3. Build results per equipment
        equipment_results = []
        for obj in tracked_objects:
            if self.pipeline_mode == "custom":
                result = self._process_custom(proc_frame, obj, now)
            else:
                result = self._process_generic(proc_frame, obj, now)
            if result:
                equipment_results.append(result)

        self.previous_frame = proc_frame

        # 4. Annotate
        annotated = self._draw_annotations(proc_frame, equipment_results)

        # 5. Summary
        summary = self.time_tracker.get_summary()

        self._last_result = {
            "frame_id": self.frame_count,
            "timestamp": now,
            "equipment": equipment_results,
            "annotated_frame": annotated,
            "summary": summary
        }
        return self._last_result

    def _process_custom(self, frame: np.ndarray, obj: Dict[str, Any], timestamp: float) -> Optional[Dict[str, Any]]:
        """
        Custom mode:
          - Excavator: Run through generic optical flow pipeline to get specific activity (Digging/Swinging).
          - Trucks: Use lightweight tracker motion to get simple state (Hauling/Idle).
        """
        tracker_id = obj["tracker_id"]
        equipment_type = obj["class_name"]

        # Drill down into excavator activity using optical flow
        if equipment_type == "Excavator":
            return self._process_generic(frame, obj, timestamp)

        # For trucks, simple centroid movement is enough
        is_active = obj.get("is_moving", False)

        if is_active:
            if equipment_type == "Dump Truck":
                activity = "Hauling"
            elif equipment_type == "Concrete Mixer":
                activity = "Transporting"
            else:
                activity = "Active"
        else:
            # Map down to generic Waiting state
            activity = "Waiting"

        # Update time tracker
        stats = self.time_tracker.update(
            tracker_id=tracker_id,
            is_active=is_active,
            activity=activity,
            timestamp=timestamp
        )

        return {
            "equipment_id": tracker_id,
            "bbox": obj["bbox"],
            "class_name": equipment_type,
            "confidence": obj["confidence"],
            "state": "ACTIVE" if is_active else "IDLE",
            "activity": activity,
            "activity_confidence": obj["confidence"],
            "algorithm": "tracker_motion",
            "motion_score": obj.get("velocity", 0.0),
            "bbox_change": obj.get("bbox_change", 0.0),
            "active_regions": [],
            "active_time": stats.active_time,
            "idle_time": stats.idle_time,
            "utilization": stats.utilization,
            "timestamp": timestamp
        }

    def _process_generic(self, frame: np.ndarray, obj: Dict[str, Any], timestamp: float) -> Optional[Dict[str, Any]]:
        """Generic mode: optical flow + classifier."""
        tracker_id = obj["tracker_id"]
        bbox = obj["bbox"]

        if self.previous_frame is not None and self.motion_analyzer:
            motion_result = self.motion_analyzer.analyze(
                frame, self.previous_frame, bbox, tracker_id
            )
        else:
            from cv_service.motion_analyzer import MotionResult
            motion_result = MotionResult()

        is_active = self.temporal_smoother.smooth(tracker_id, motion_result.is_active)

        feature_seq = self.feature_extractor.extract(
            tracker_id, bbox,
            motion_result.region_scores,
            motion_result.flow_directions
        )
        video_clip = self.clip_buffer.add_frame(tracker_id, frame, bbox) if hasattr(self, 'clip_buffer') else None

        classification = self.activity_classifier.classify(
            tracker_id=tracker_id,
            region_scores=motion_result.region_scores,
            flow_directions=motion_result.flow_directions,
            is_active=is_active,
            feature_sequence=feature_seq,
            video_clip=video_clip
        )

        stats = self.time_tracker.update(
            tracker_id=tracker_id,
            is_active=is_active,
            activity=classification["activity"],
            timestamp=timestamp
        )

        return {
            "equipment_id": tracker_id,
            "bbox": bbox,
            "class_name": obj["class_name"],
            "confidence": obj["confidence"],
            "state": "ACTIVE" if is_active else "IDLE",
            "activity": classification["activity"],
            "activity_confidence": classification["confidence"],
            "algorithm": classification["algorithm"],
            "motion_score": motion_result.motion_score,
            "bbox_change": 0.0,
            "active_regions": motion_result.active_regions,
            "active_time": stats.active_time,
            "idle_time": stats.idle_time,
            "utilization": stats.utilization,
            "timestamp": timestamp
        }

    def _draw_annotations(self, frame: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes, equipment labels, and state on frame."""
        annotated = frame.copy()

        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            state = r["state"]
            equipment_type = r["class_name"]
            activity = r["activity"]

            # Color based on equipment type, border based on state
            eq_color = EQUIPMENT_COLORS.get(equipment_type, (200, 200, 200))
            state_color = STATE_COLORS.get(state, (200, 200, 200))

            # Thick box in equipment color
            cv2.rectangle(annotated, (x1, y1), (x2, y2), eq_color, 3)
            # Thin inner box showing state
            cv2.rectangle(annotated, (x1+2, y1+2), (x2-2, y2-2), state_color, 1)

            # State indicator dot
            dot_color = (0, 255, 0) if state == "ACTIVE" else (0, 0, 255)
            cv2.circle(annotated, (x2 - 12, y1 + 12), 6, dot_color, -1)
            cv2.circle(annotated, (x2 - 12, y1 + 12), 7, (255, 255, 255), 1)

            # Labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2

            line1 = f"#{r['equipment_id']} {equipment_type}"
            line2 = f"{activity} | {r['confidence']:.0%}"
            line3 = f"Util: {r['utilization']:.0%}"

            labels = [line1, line2, line3]
            for i, text in enumerate(labels):
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                ly = y1 - 8 - (len(labels) - 1 - i) * (th + 8)
                cv2.rectangle(annotated, (x1, ly - th - 4), (x1 + tw + 8, ly + 4), eq_color, -1)
                cv2.putText(annotated, text, (x1 + 4, ly), font, font_scale, (0, 0, 0), thickness)

        # Top-left overlay
        active_count = sum(1 for r in results if r["state"] == "ACTIVE")
        idle_count = len(results) - active_count
        info_text = f"Frame: {self.frame_count} | Equipment: {len(results)} | Active: {active_count} | Idle: {idle_count}"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated

    def process_video(self, source: str, max_frames: int = 0) -> Generator[Dict[str, Any], None, None]:
        """Process a video source frame by frame."""
        try:
            source_input = int(source)
        except (ValueError, TypeError):
            source_input = source

        cap = cv2.VideoCapture(source_input)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.time_tracker.set_fps(self.fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video opened: {source}")
        logger.info(f"  FPS: {self.fps}, Total frames: {total_frames}")
        logger.info(f"  Frame skip: {self.frame_skip}, Process width: {self.process_width}px")
        logger.info(f"  Pipeline mode: {self.pipeline_mode}")

        frame_interval = 1.0 / self.fps
        start_time = time.time()
        raw_frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                raw_frame_idx += 1
                timestamp = start_time + (raw_frame_idx * frame_interval)

                if self.frame_skip > 1 and raw_frame_idx % self.frame_skip != 0:
                    if self._last_result is not None:
                        proc_frame = self._downscale(frame)
                        annotated = self._draw_annotations(proc_frame, self._last_result["equipment"])
                        skipped_result = self._last_result.copy()
                        skipped_result["annotated_frame"] = annotated
                        skipped_result["timestamp"] = timestamp
                        yield skipped_result
                        continue

                result = self.process_frame(frame, timestamp)
                yield result

                if max_frames > 0 and self.frame_count >= max_frames:
                    break

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        finally:
            cap.release()
            effective_fps = self.frame_count / max(time.time() - start_time, 0.001)
            logger.info(
                f"Video processing complete. "
                f"{self.frame_count} frames processed / {raw_frame_idx} total. "
                f"Effective: {effective_fps:.1f} FPS"
            )

    def reset(self):
        """Reset pipeline state."""
        self.previous_frame = None
        self.frame_count = 0
        self.time_tracker.reset()
        self.tracker.reset()
