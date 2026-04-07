"""
Hybrid Activity Classifier.

Combines rule-based heuristics with learned models (LSTM, C3D) for
robust activity classification of construction equipment.

Activities: Digging, Swinging/Loading, Dumping, Waiting

Classification Strategy:
    1. Rule-based heuristics (always available, fast fallback)
    2. LSTM classifier (primary when loaded, temporal patterns)
    3. C3D classifier (optional, GPU, spatiotemporal features)
    
The hybrid approach uses LSTM as primary, falling back to rules
when confidence is below threshold.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

ACTIVITY_CLASSES = ["Digging", "Swinging/Loading", "Dumping", "Waiting"]
ACTIVITY_IDX = {name: i for i, name in enumerate(ACTIVITY_CLASSES)}


class RuleBasedClassifier:
    """
    Rule-based activity classifier using optical flow region patterns.
    
    Region grid layout (3x3):
        [0,0] [0,1] [0,2]    ← Top (bucket/arm tip)
        [1,0] [1,1] [1,2]    ← Middle (arm body)
        [2,0] [2,1] [2,2]    ← Bottom (cab/tracks)
    
    Heuristics:
        - Digging: High motion in top regions, low in bottom, downward flow
        - Swinging/Loading: Lateral motion across columns, rotation pattern
        - Dumping: Top region motion with upward flow trajectory
        - Waiting: No significant motion in any region
    """

    def __init__(self, motion_threshold: float = 2.5):
        self.threshold = motion_threshold

    def classify(
        self,
        region_scores: np.ndarray,
        flow_directions: np.ndarray,
        is_active: bool
    ) -> Tuple[str, float]:
        """
        Classify activity based on region motion patterns.

        Args:
            region_scores: (3, 3) motion magnitude per region
            flow_directions: (3, 3, 2) flow direction vectors
            is_active: Whether equipment is currently active

        Returns:
            (activity_label, confidence)
        """
        if not is_active:
            return "Waiting", 0.95

        # Region aggregations
        top_motion = float(np.mean(region_scores[0, :]))
        mid_motion = float(np.mean(region_scores[1, :]))
        bot_motion = float(np.mean(region_scores[2, :]))
        left_motion = float(np.mean(region_scores[:, 0]))
        right_motion = float(np.mean(region_scores[:, 2]))
        overall_motion = float(np.mean(region_scores))

        # Flow direction analysis
        top_flow_y = float(np.mean(flow_directions[0, :, 1]))  # Vertical component
        lateral_spread = abs(left_motion - right_motion)

        # Mean horizontal flow across all regions
        mean_flow_x = float(np.mean(np.abs(flow_directions[:, :, 0])))
        mean_flow_y = float(np.mean(flow_directions[:, :, 1]))

        scores = {
            "Digging": 0.0,
            "Swinging/Loading": 0.0,
            "Dumping": 0.0,
            "Waiting": 0.0
        }

        # --- Digging Detection ---
        # Arm moves down: high top/mid motion, downward flow, low base motion
        if top_motion > self.threshold and bot_motion < top_motion * 0.6:
            scores["Digging"] += 0.4
        if top_flow_y > 0.5:  # Downward flow
            scores["Digging"] += 0.3
        if mid_motion > self.threshold * 0.8:
            scores["Digging"] += 0.2

        # --- Swinging / Loading Detection ---
        # Lateral motion pattern: significant horizontal flow, rotation
        if mean_flow_x > self.threshold * 0.8:
            scores["Swinging/Loading"] += 0.4
        if lateral_spread > self.threshold * 0.5:
            scores["Swinging/Loading"] += 0.3
        # Whole body motion (cab rotating)
        if bot_motion > self.threshold * 0.7 and mid_motion > self.threshold * 0.7:
            scores["Swinging/Loading"] += 0.2

        # --- Dumping Detection ---
        # Top regions active, upward flow trajectory
        if top_motion > self.threshold and top_flow_y < -0.5:  # Upward flow
            scores["Dumping"] += 0.5
        if top_motion > mid_motion and top_motion > bot_motion:
            scores["Dumping"] += 0.2

        # --- Waiting (active but low meaningful motion) ---
        if overall_motion < self.threshold * 0.5:
            scores["Waiting"] += 0.6

        # Normalize and select best
        total = sum(scores.values())
        if total == 0:
            return "Waiting", 0.5

        for key in scores:
            scores[key] /= total

        best_activity = max(scores, key=scores.get)
        confidence = scores[best_activity]

        return best_activity, confidence


class HybridActivityClassifier:
    """
    Hybrid classifier combining rule-based, LSTM, and C3D approaches.
    
    Priority:
        1. LSTM (if model loaded and confidence >= threshold)
        2. C3D (if model loaded, GPU available, and confidence >= threshold)
        3. Rule-based heuristics (always available)
    """

    def __init__(self, config, lstm_model=None, c3d_model=None):
        self.config = config
        self.rule_classifier = RuleBasedClassifier(config.threshold)
        self.lstm_model = lstm_model
        self.c3d_model = c3d_model
        self.confidence_threshold = getattr(config, 'active_zone_threshold', 0.6)

        # Temporal smoothing per tracker
        self._histories: Dict[int, deque] = {}
        self._smoothing_window = config.temporal_smoothing_frames

        # Track which algorithm was used (for logging/dashboard)
        self.last_algorithm_used: Dict[int, str] = {}

    def classify(
        self,
        tracker_id: int,
        region_scores: np.ndarray,
        flow_directions: np.ndarray,
        is_active: bool,
        feature_sequence: Optional[np.ndarray] = None,
        video_clip: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Classify activity using the best available algorithm.

        Args:
            tracker_id: Equipment tracker ID
            region_scores: (3, 3) motion magnitude per region
            flow_directions: (3, 3, 2) flow direction vectors
            is_active: Whether equipment is active
            feature_sequence: Optional (seq_len, 27) for LSTM
            video_clip: Optional (16, 3, 112, 112) for C3D

        Returns:
            Dict with activity, confidence, algorithm_used, probabilities
        """
        activity = "Waiting"
        confidence = 0.0
        probabilities = [0.0] * len(ACTIVITY_CLASSES)
        algorithm_used = "rule_based"

        # Try LSTM first (best for temporal patterns)
        if (
            self.lstm_model is not None
            and feature_sequence is not None
            and self.config.activity_algorithm in ("lstm", "hybrid")
        ):
            lstm_activity, lstm_conf, lstm_probs = self.lstm_model.predict(feature_sequence)
            if lstm_conf >= self.confidence_threshold:
                activity = lstm_activity
                confidence = lstm_conf
                probabilities = lstm_probs
                algorithm_used = "lstm"

        # Try C3D if LSTM didn't meet threshold
        if (
            algorithm_used == "rule_based"
            and self.c3d_model is not None
            and video_clip is not None
            and self.config.activity_algorithm in ("c3d", "hybrid")
        ):
            c3d_activity, c3d_conf, c3d_probs = self.c3d_model.predict(video_clip)
            if c3d_conf >= self.confidence_threshold:
                activity = c3d_activity
                confidence = c3d_conf
                probabilities = c3d_probs
                algorithm_used = "c3d"

        # Fall back to rule-based
        if algorithm_used == "rule_based":
            activity, confidence = self.rule_classifier.classify(
                region_scores, flow_directions, is_active
            )
            # Convert to probabilities
            probabilities = [0.0] * len(ACTIVITY_CLASSES)
            idx = ACTIVITY_IDX.get(activity, 3)
            probabilities[idx] = confidence
            remaining = (1.0 - confidence) / max(len(ACTIVITY_CLASSES) - 1, 1)
            for i in range(len(probabilities)):
                if i != idx:
                    probabilities[i] = remaining

        # Apply temporal smoothing
        activity = self._temporal_smooth(tracker_id, activity)
        self.last_algorithm_used[tracker_id] = algorithm_used

        return {
            "activity": activity,
            "confidence": round(confidence, 3),
            "probabilities": {
                ACTIVITY_CLASSES[i]: round(p, 3)
                for i, p in enumerate(probabilities)
            },
            "algorithm": algorithm_used
        }

    def _temporal_smooth(self, tracker_id: int, activity: str) -> str:
        """Apply majority vote smoothing over recent predictions."""
        if tracker_id not in self._histories:
            self._histories[tracker_id] = deque(maxlen=self._smoothing_window)

        self._histories[tracker_id].append(activity)
        history = self._histories[tracker_id]

        # Majority vote
        counts = {}
        for a in history:
            counts[a] = counts.get(a, 0) + 1

        return max(counts, key=counts.get)
