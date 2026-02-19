import torch
import torch.nn as nn


class AnomalyLoss(nn.Module):
    """BCE loss between predicted anomaly scores and target probabilities."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(
        self, predicted_scores: torch.Tensor, target_probabilities: torch.Tensor
    ) -> torch.Tensor:
        return self.bce(predicted_scores, target_probabilities)
