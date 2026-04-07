"""
LSTM Model for Activity Classification.

Bidirectional LSTM that operates on temporal sequences of per-frame
feature vectors extracted from tracked equipment objects.

Input: (batch, seq_len=30, features=27)
Output: (batch, 4) — activity class probabilities
Classes: Digging, Swinging/Loading, Dumping, Waiting
"""

import logging
import os
from typing import Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

# Activity class labels
ACTIVITY_CLASSES = ["Digging", "Swinging/Loading", "Dumping", "Waiting"]


class LSTMActivityModel:
    """
    Bidirectional LSTM for temporal activity classification.
    
    Architecture:
        Input (27-dim) → BiLSTM (2 layers, 128 hidden) → Dropout → FC → 4 classes
    
    Can be used in two modes:
        1. Pretrained: Load saved weights
        2. Self-supervised: Train on rule-based labels, then generalize
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = "cpu"
        self._build_model()

    def _build_model(self):
        """Build PyTorch LSTM model."""
        try:
            import torch
            import torch.nn as nn

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            class BiLSTMClassifier(nn.Module):
                def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_dim,
                        hidden_size=hidden_dim,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=True,
                        dropout=dropout if num_layers > 1 else 0
                    )
                    self.dropout = nn.Dropout(dropout)
                    self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

                def forward(self, x):
                    # x: (batch, seq_len, features)
                    lstm_out, _ = self.lstm(x)
                    # Take the last timestep output
                    last_out = lstm_out[:, -1, :]
                    out = self.dropout(last_out)
                    logits = self.fc(out)
                    return logits

            self.model = BiLSTMClassifier(
                input_dim=self.config.feature_dim,
                hidden_dim=self.config.hidden_size,
                num_layers=self.config.num_layers,
                num_classes=self.config.num_classes,
                dropout=self.config.dropout
            ).to(self.device)

            self.model.eval()
            self._torch = torch
            self._nn = nn

            logger.info(f"LSTM model built on {self.device}")
            logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        except ImportError:
            logger.warning("PyTorch not available — LSTM model disabled")
            self.model = None

    def predict(self, sequence: np.ndarray) -> Tuple[str, float, List[float]]:
        """
        Predict activity from a feature sequence.

        Args:
            sequence: (seq_len, feature_dim) numpy array

        Returns:
            Tuple of (activity_label, confidence, all_probabilities)
        """
        if self.model is None:
            return "Waiting", 0.0, [0.0] * len(ACTIVITY_CLASSES)

        import torch

        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = ACTIVITY_CLASSES[pred_idx]

        return label, confidence, probs.tolist()

    def self_supervised_train(
        self,
        sequences: List[np.ndarray],
        labels: List[int],
        epochs: int = 20,
        lr: float = 0.001
    ):
        """
        Train the LSTM on rule-based generated labels (self-supervised).
        
        Args:
            sequences: List of (seq_len, feature_dim) arrays
            labels: List of class indices (0-3)
            epochs: Training epochs
            lr: Learning rate
        """
        if self.model is None:
            logger.warning("Cannot train — model not built")
            return

        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        X = torch.FloatTensor(np.array(sequences)).to(self.device)
        y = torch.LongTensor(np.array(labels)).to(self.device)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total += len(batch_y)

            if (epoch + 1) % 5 == 0:
                acc = correct / max(total, 1)
                logger.info(f"LSTM Training Epoch {epoch+1}/{epochs} — Loss: {total_loss:.4f}, Acc: {acc:.3f}")

        self.model.eval()

    def save(self, path: str):
        """Save model weights."""
        if self.model is not None:
            import torch
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)
            logger.info(f"LSTM model saved to {path}")

    def load(self, path: str) -> bool:
        """Load model weights. Returns True if successful."""
        if self.model is None:
            return False
        try:
            import torch
            self.model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            logger.info(f"LSTM model loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load LSTM model: {e}")
            return False
