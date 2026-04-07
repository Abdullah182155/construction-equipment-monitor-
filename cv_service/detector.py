"""
YOLOv8 Object Detector for Construction Equipment.

Supports two modes:
  - Custom model (best.pt): Classes are activities (Digging, Swing, etc.)
  - Generic model (yolov8n.pt): COCO classes filtered by size

The custom model directly classifies equipment activity, eliminating
the need for separate optical flow and classifier stages.
"""

import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class EquipmentDetector:
    """Detects construction equipment in video frames using YOLOv8."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.config.model_path)
            logger.info(f"YOLOv8 model loaded: {self.config.model_path}")
            # Log class names from model
            if hasattr(self.model, 'names'):
                logger.info(f"  Model classes: {self.model.names}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a single frame.

        Returns list of detections with bbox, confidence, class_id, class_name.
        class_name directly indicates the activity for custom-trained models.
        """
        if self.model is None:
            return []

        results = self.model(
            frame,
            conf=self.config.confidence,
            verbose=False,
            imgsz=640
        )

        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Get class name from model or config
                if hasattr(self.model, 'names') and cls_id in self.model.names:
                    cls_name = self.model.names[cls_id]
                else:
                    cls_name = self.config.class_names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name
                })

        logger.debug(f"Detected {len(detections)} objects")
        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Run detection on a batch of frames."""
        return [self.detect(frame) for frame in frames]
