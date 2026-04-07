"""
Centralized configuration for the Equipment Monitoring System.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class YOLOConfig:
    """YOLO object detection configuration."""
    model_path: str = os.getenv("YOLO_MODEL", "best.pt")
    confidence: float = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
    target_classes: List[int] = field(default_factory=lambda: [
        int(c) for c in os.getenv("YOLO_TARGET_CLASSES", "0,1,2").split(",")
    ])
    # Custom model class name mapping
    # 0=concrete_mixer_truck, 1=dump_truck, 2=excavator
    class_names: dict = field(default_factory=lambda: {
        0: "Concrete Mixer",
        1: "Dump Truck",
        2: "Excavator"
    })


@dataclass
class MotionConfig:
    """Motion analysis configuration."""
    threshold: float = float(os.getenv("MOTION_THRESHOLD", "2.5"))
    grid_rows: int = int(os.getenv("MOTION_GRID_ROWS", "3"))
    grid_cols: int = int(os.getenv("MOTION_GRID_COLS", "3"))
    active_zone_threshold: int = int(os.getenv("ACTIVE_ZONE_THRESHOLD", "1"))
    temporal_smoothing_frames: int = int(os.getenv("TEMPORAL_SMOOTHING_FRAMES", "5"))
    algorithm: str = os.getenv("MOTION_ALGORITHM", "optical_flow")
    activity_algorithm: str = os.getenv("ACTIVITY_ALGORITHM", "hybrid")


@dataclass
class LSTMConfig:
    """LSTM model configuration."""
    hidden_size: int = int(os.getenv("LSTM_HIDDEN_SIZE", "128"))
    num_layers: int = int(os.getenv("LSTM_NUM_LAYERS", "2"))
    sequence_length: int = int(os.getenv("LSTM_SEQUENCE_LENGTH", "30"))
    confidence_threshold: float = float(os.getenv("LSTM_CONFIDENCE_THRESHOLD", "0.6"))
    dropout: float = 0.3
    num_classes: int = 4  # Digging, Swinging/Loading, Dumping, Waiting
    feature_dim: int = 27  # 9 zones * 2 (mag+dir) + 9 (bbox+kinematics)


@dataclass
class C3DConfig:
    """C3D model configuration."""
    clip_length: int = int(os.getenv("C3D_CLIP_LENGTH", "16"))
    feature_dim: int = int(os.getenv("C3D_FEATURE_DIM", "4096"))
    crop_size: int = 112
    num_classes: int = 4


@dataclass
class KafkaConfig:
    """Kafka messaging configuration."""
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic: str = os.getenv("KAFKA_TOPIC", "equipment-events")
    group_id: str = os.getenv("KAFKA_GROUP_ID", "equipment-consumer-group")


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration."""
    url: str = os.getenv("DATABASE_URL",
                         "postgresql://equipment:equipment123@localhost:5432/equipment_monitoring")


@dataclass
class APIConfig:
    """API service configuration."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    ws_frame_interval: float = float(os.getenv("WS_FRAME_INTERVAL", "0.033"))
    ws_jpeg_quality: int = int(os.getenv("WS_JPEG_QUALITY", "60"))


@dataclass
class VideoConfig:
    """Video source configuration."""
    source: str = os.getenv("VIDEO_SOURCE", "data/sample_video/excavator_demo.mp4")
    frame_skip: int = int(os.getenv("FRAME_SKIP", "2"))  # Process every Nth frame
    process_width: int = int(os.getenv("PROCESS_WIDTH", "640"))  # Downscale for YOLO


@dataclass
class Settings:
    """Master configuration container."""
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    c3d: C3DConfig = field(default_factory=C3DConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    pipeline_mode: str = os.getenv("PIPELINE_MODE", "custom")  # "custom" or "generic"


# Global settings singleton
settings = Settings()
