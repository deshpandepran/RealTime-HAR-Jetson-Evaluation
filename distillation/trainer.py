import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class KnowledgeDistillationTrainer:
    """Manages training, evaluation, and DDP configuration for distilling

    knowledge from the frozen TV-L1 teacher to the IMFE student.
    """

    def __init__(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        use_ddp: bool = False,
    ):
        self.device = device
        self.use_ddp = use_ddp

        # Teacher remains frozen
        self.teacher = teacher_model.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Student model setup
        self.student = student_model.to(self.device)
        if self.use_ddp:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.student = DDP(
                self.student, device_ids=[local_rank], output_device=local_rank
            )

        self.criterion = criterion
        self.optimizer = optimizer

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Performs a single training epoch using feature distillation."""
        self.student.train()
        total_loss = 0.0

        for batch_idx, batch_data in enumerate(dataloader):
            # Unpack clip frames (expecting shape [B, K, C, H, W])
            frames = batch_data["frames"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass through frozen teacher (TV-L1 flow + ResNet-50 feature extractor)
            with torch.no_grad():
                teacher_feats = self.teacher(frames)

            # Forward pass through IMFE student
            student_feats = self.student(frames)

            # Compute MSE distillation loss
            loss = self.criterion(student_feats, teacher_feats)

            # Backpropagation & optimization
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    @staticmethod
    def setup_ddp():
        """Initializes Distributed Data Parallel environment."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    @staticmethod
    def cleanup_ddp():
        """Cleans up DDP process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
