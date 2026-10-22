import torch
import torch.nn as nn


class FeatureDistillationLoss(nn.Module):
    """Computes Mean Squared Error (MSE) loss between student IMFE feature representations

    and frozen teacher (TV-L1 + ResNet-50) feature representations.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.criterion = nn.MSELoss(reduction=reduction)

    def forward(
        self, student_features: torch.Tensor, teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the distillation loss.

        Args:
            student_features (torch.Tensor): Output features from IMFE [B, D].
            teacher_features (torch.Tensor): Output features from frozen teacher
              [B, D].

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if student_features.shape != teacher_features.shape:
            raise ValueError(
                f"Shape mismatch: student features {student_features.shape} "
                f"must match teacher features {teacher_features.shape}."
            )
        return self.criterion(student_features, teacher_features.detach())
      
