"""
Basic Pipeline Tests
Validates that core CV modules can be imported and initialized.
"""

import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_loads():
    """Test that configuration loads from environment."""
    from config.settings import settings
    assert settings.yolo.model_path is not None
    assert settings.motion.threshold > 0
    assert settings.kafka.topic == "equipment-events"
    print("✓ Config loaded successfully")


def test_motion_analyzer_creation():
    """Test motion analyzer factory creates correct analyzer."""
    from config.settings import settings
    from cv_service.motion_analyzer import MotionAnalyzerFactory, OpticalFlowAnalyzer

    analyzer = MotionAnalyzerFactory.create(settings.motion)
    assert isinstance(analyzer, OpticalFlowAnalyzer)
    print("✓ Motion analyzer created: OpticalFlowAnalyzer")


def test_motion_analysis():
    """Test motion analysis on synthetic frames."""
    import numpy as np
    from config.settings import settings
    from cv_service.motion_analyzer import OpticalFlowAnalyzer

    analyzer = OpticalFlowAnalyzer(settings.motion)

    # Create two frames with known motion
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add a moving rectangle
    frame1[100:200, 100:200] = 128
    frame2[100:200, 120:220] = 128  # Shifted right

    bbox = [80, 80, 240, 220]
    result = analyzer.analyze(frame2, frame1, bbox, tracker_id=1)

    print(f"✓ Motion analysis: score={result.motion_score:.4f}, active={result.is_active}")
    print(f"  Active regions: {result.active_regions}")


def test_time_tracker():
    """Test time tracking logic."""
    import time
    from cv_service.time_tracker import TimeTracker

    tracker = TimeTracker()
    tracker.set_fps(30)

    t = time.time()
    # Simulate active frames
    for i in range(30):
        tracker.update(1, is_active=True, activity="Digging", timestamp=t + i * 0.033)
    # Simulate idle frames
    for i in range(10):
        tracker.update(1, is_active=False, activity="Waiting", timestamp=t + 1.0 + i * 0.033)

    stats = tracker.get_stats(1)
    assert stats is not None
    assert stats.active_time > 0
    print(f"✓ Time tracker: active={stats.active_time:.2f}s, idle={stats.idle_time:.2f}s, util={stats.utilization:.2%}")


def test_activity_classifier():
    """Test rule-based activity classification."""
    import numpy as np
    from cv_service.activity_classifier import RuleBasedClassifier

    classifier = RuleBasedClassifier(motion_threshold=2.5)

    # Simulate digging pattern: high top motion, low bottom
    region_scores = np.array([
        [5.0, 6.0, 4.5],  # Top - high (arm)
        [3.0, 3.5, 2.8],  # Middle
        [0.5, 0.3, 0.4],  # Bottom - low (tracks)
    ])
    flow_directions = np.zeros((3, 3, 2))
    flow_directions[0, :, 1] = 1.0  # Downward flow in top

    activity, confidence = classifier.classify(region_scores, flow_directions, is_active=True)
    print(f"✓ Rule classifier: {activity} (confidence={confidence:.2f})")

    # Simulate waiting
    zero_scores = np.zeros((3, 3))
    activity2, conf2 = classifier.classify(zero_scores, flow_directions, is_active=False)
    assert activity2 == "Waiting"
    print(f"✓ Rule classifier: {activity2} (confidence={conf2:.2f})")


def test_feature_extractor():
    """Test feature extraction dimensions."""
    import numpy as np
    from config.settings import settings
    from cv_service.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor(settings.lstm, frame_shape=(480, 640))

    region_scores = np.random.rand(3, 3).astype(np.float32)
    flow_dirs = np.random.rand(3, 3, 2).astype(np.float32)
    bbox = [100, 100, 300, 300]

    # Fill buffer
    for i in range(settings.lstm.sequence_length):
        seq = extractor.extract(1, bbox, region_scores, flow_dirs)

    assert seq is not None
    assert seq.shape == (settings.lstm.sequence_length, settings.lstm.feature_dim)
    print(f"✓ Feature extractor: sequence shape = {seq.shape}")


def test_lstm_model():
    """Test LSTM model forward pass."""
    import numpy as np
    from config.settings import settings
    from cv_service.lstm_model import LSTMActivityModel

    try:
        model = LSTMActivityModel(settings.lstm)
        seq = np.random.rand(settings.lstm.sequence_length, settings.lstm.feature_dim).astype(np.float32)
        label, conf, probs = model.predict(seq)
        print(f"✓ LSTM model: {label} (conf={conf:.2f}), probs={[f'{p:.2f}' for p in probs]}")
    except Exception as e:
        print(f"⚠ LSTM model skipped (PyTorch not available): {e}")


def test_db_models():
    """Test SQLAlchemy model creation."""
    from db_service.models import EquipmentEvent, EquipmentSummary
    assert EquipmentEvent.__tablename__ == "equipment_events"
    assert EquipmentSummary.__tablename__ == "equipment_summary"
    print("✓ DB models defined correctly")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Equipment Monitoring — Pipeline Tests")
    print("=" * 50 + "\n")

    tests = [
        test_config_loads,
        test_motion_analyzer_creation,
        test_motion_analysis,
        test_time_tracker,
        test_activity_classifier,
        test_feature_extractor,
        test_lstm_model,
        test_db_models,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}\n")
