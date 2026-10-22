"""Combined MSE + cosine-similarity loss for IMFE knowledge distillation.

Part of Option C in issue #10 (Technical Extensions & Optimization).

    loss = MSE(student, teacher) + lam * mean(1 - cos(student, teacher))

lam = 0 reduces to the MSE-only loss used in the paper, so it can be used as
the baseline in the sweep.

Usage (once the distillation loop from issue #6 exists):
    from extensions.losses import CosineMSELoss
    criterion = CosineMSELoss(lam=0.1)
    loss = criterion(student_feats, teacher_feats)   # both (B, 2048)
    loss.backward()

Self-test (runs without the Jetson or any trained weights):
    python extensions/losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineMSELoss(nn.Module):
    """MSE plus a weighted cosine-direction term.

    Args:
        lam: Weight of the cosine term. 0 gives plain MSE.
    """

    def __init__(self, lam: float = 1.0):
        super().__init__()
        if lam < 0:
            raise ValueError(f"lam must be >= 0, got {lam}")
        self.lam = lam
        self.mse = nn.MSELoss()

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Return the combined scalar loss."""
        total, _ = self.forward_with_parts(student, teacher)
        return total

    def forward_with_parts(self, student: torch.Tensor, teacher: torch.Tensor):
        """Return (total_loss, {"mse": ..., "cos_term": ...}).

        Use this during training to log each term, since MSE depends on feature
        scale and the cosine term does not. Both terms are needed to choose lam.
        """
        if student.shape != teacher.shape:
            raise ValueError(
                f"shape mismatch: student {tuple(student.shape)} vs teacher {tuple(teacher.shape)}"
            )

        # The teacher is frozen (TV-L1 + ResNet-50), so no gradient flows into it.
        teacher = teacher.detach()

        mse = self.mse(student, teacher)
        cos = F.cosine_similarity(student, teacher, dim=1)  # shape (B,)
        cos_term = (1.0 - cos).mean()

        total = mse + self.lam * cos_term
        return total, {"mse": mse.detach(), "cos_term": cos_term.detach()}


def _self_test() -> None:
    torch.manual_seed(0)
    criterion_mse_only = CosineMSELoss(lam=0.0)
    criterion_combined = CosineMSELoss(lam=1.0)

    # 1. Identical inputs: both terms should be zero.
       # 1. Identical inputs: both terms should be (numerically) zero.
    x = torch.randn(4, 2048)
    total, parts = criterion_combined.forward_with_parts(x, x.clone())
    assert abs(total.item()) < 1e-6, f"identical inputs should give ~0, got {total.item()}"
    assert abs(parts["cos_term"].item()) < 1e-6

    # 2. Opposite-direction inputs: cosine term should be 2.
    total, parts = criterion_combined.forward_with_parts(-x, x)
    assert abs(parts["cos_term"].item() - 2.0) < 1e-5, parts["cos_term"]

    # 3. lam = 0 must equal plain MSE.
    s, t = torch.randn(4, 2048), torch.randn(4, 2048)
    assert torch.allclose(
        criterion_mse_only(s, t), nn.functional.mse_loss(s, t)
    ), "lam=0 should equal MSE"

    # 4. Gradients reach the student but not the teacher.
    s = torch.randn(4, 2048, requires_grad=True)
    t = torch.randn(4, 2048, requires_grad=True)
    criterion_combined(s, t).backward()
    assert s.grad is not None and s.grad.abs().sum() > 0, "student should get gradient"
    assert t.grad is None, "teacher should not get gradient"

    # 5. Shape mismatch is rejected.
    try:
        criterion_combined(torch.randn(4, 2048), torch.randn(4, 1024))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch should raise ValueError")
        # 6. Output must be a scalar.
    out = criterion_combined(torch.randn(4, 2048), torch.randn(4, 2048))
    assert out.dim() == 0, f"loss should be a scalar, got shape {tuple(out.shape)}"

    print("All CosineMSELoss self-tests passed.")


if __name__ == "__main__":
    _self_test()