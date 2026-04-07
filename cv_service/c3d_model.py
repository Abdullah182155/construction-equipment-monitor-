"""
C3D (3D Convolutional Network) for Spatiotemporal Feature Extraction.

Operates on 16-frame video clips cropped from tracked bounding boxes.
Uses pretrained weights (Sports-1M) as feature extractor, with a
lightweight classification head for activity recognition.

Input: (batch, 3, 16, 112, 112) — clip of 16 RGB frames at 112x112
Output: Activity class probabilities or 4096-dim feature vector
"""

import logging
import os
from typing import Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

ACTIVITY_CLASSES = ["Digging", "Swinging/Loading", "Dumping", "Waiting"]


class C3DModel:
    """
    C3D spatiotemporal feature extractor with classification head.
    
    Architecture:
        Conv3D layers (8 conv + 5 pool) → FC6 (4096) → FC7 (4096) → Classifier (4)
    
    Modes:
        - Feature extraction: Returns fc6 4096-dim vector
        - Classification: Returns activity class probabilities
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.classifier = None
        self.device = "cpu"
        self._build_model()

    def _build_model(self):
        """Build C3D architecture in PyTorch."""
        try:
            import torch
            import torch.nn as nn

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            class C3DNetwork(nn.Module):
                """C3D network architecture (Tran et al. 2015)."""
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                    self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                    self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                    self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                    self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                    self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                    self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                    self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                    self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                    self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                    self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                    self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                    self.pool5 = nn.AdaptiveAvgPool3d((1, 1, 1))

                    self.fc6 = nn.Linear(512, 4096)
                    self.fc7 = nn.Linear(4096, 4096)
                    self.dropout = nn.Dropout(0.5)
                    self.relu = nn.ReLU()

                def forward(self, x, extract_features=False):
                    # x: (batch, 3, 16, 112, 112)
                    x = self.relu(self.conv1(x))
                    x = self.pool1(x)
                    x = self.relu(self.conv2(x))
                    x = self.pool2(x)
                    x = self.relu(self.conv3a(x))
                    x = self.relu(self.conv3b(x))
                    x = self.pool3(x)
                    x = self.relu(self.conv4a(x))
                    x = self.relu(self.conv4b(x))
                    x = self.pool4(x)
                    x = self.relu(self.conv5a(x))
                    x = self.relu(self.conv5b(x))
                    x = self.pool5(x)

                    x = x.view(x.size(0), -1)
                    x = self.relu(self.fc6(x))
                    fc6_features = x
                    x = self.dropout(x)
                    x = self.relu(self.fc7(x))
                    x = self.dropout(x)

                    if extract_features:
                        return fc6_features
                    return x

            # Classification head
            class C3DClassifier(nn.Module):
                def __init__(self, feature_dim=4096, num_classes=4):
                    super().__init__()
                    self.fc = nn.Sequential(
                        nn.Linear(feature_dim, 512),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(512, num_classes)
                    )

                def forward(self, x):
                    return self.fc(x)

            self.model = C3DNetwork().to(self.device)
            self.classifier = C3DClassifier(
                feature_dim=self.config.feature_dim,
                num_classes=self.config.num_classes
            ).to(self.device)

            self.model.eval()
            self.classifier.eval()
            self._torch = torch

            total_params = sum(p.numel() for p in self.model.parameters())
            total_params += sum(p.numel() for p in self.classifier.parameters())
            logger.info(f"C3D model built on {self.device}")
            logger.info(f"  Parameters: {total_params:,}")

        except ImportError:
            logger.warning("PyTorch not available — C3D model disabled")
            self.model = None

    def extract_features(self, clip: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 4096-dim feature vector from a video clip.

        Args:
            clip: (16, 3, 112, 112) numpy array

        Returns:
            (4096,) feature vector or None
        """
        if self.model is None:
            return None

        import torch

        self.model.eval()
        with torch.no_grad():
            # clip shape: (T, C, H, W) -> (1, C, T, H, W)
            x = torch.FloatTensor(clip).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
            features = self.model(x, extract_features=True)
            return features.cpu().numpy()[0]

    def predict(self, clip: np.ndarray) -> Tuple[str, float, List[float]]:
        """
        Predict activity class from a video clip.

        Args:
            clip: (16, 3, 112, 112) numpy array

        Returns:
            Tuple of (activity_label, confidence, all_probabilities)
        """
        if self.model is None or self.classifier is None:
            return "Waiting", 0.0, [0.0] * len(ACTIVITY_CLASSES)

        import torch

        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            x = torch.FloatTensor(clip).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
            features = self.model(x, extract_features=True)
            logits = self.classifier(features)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = ACTIVITY_CLASSES[pred_idx]

        return label, confidence, probs.tolist()

    def save(self, path: str):
        """Save model weights."""
        if self.model is not None:
            import torch
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            torch.save({
                "c3d": self.model.state_dict(),
                "classifier": self.classifier.state_dict()
            }, path)
            logger.info(f"C3D model saved to {path}")

    def load(self, path: str) -> bool:
        """Load model weights."""
        if self.model is None:
            return False
        try:
            import torch
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["c3d"])
            self.classifier.load_state_dict(checkpoint["classifier"])
            self.model.eval()
            self.classifier.eval()
            logger.info(f"C3D model loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load C3D model: {e}")
            return False
