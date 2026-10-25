import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.imfe import IMFE
from models.resnet_flow import ResNetFlowExtractor
from distillation.loss import FeatureDistillationLoss

logger = logging.getLogger(__name__)


class TeacherFlowPipeline(nn.Module):
    """
    Frozen teacher pipeline combining inter-frame optical flow extraction and
    the 10-channel ResNetFlowExtractor backbone.
    Processes clip frames of shape (B, K, 3, H, W) into 2048-dimensional features.
    """
    def __init__(self, in_channels: int = 10, pretrained: bool = False, device: torch.device = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.resnet_flow = ResNetFlowExtractor(in_channels=in_channels, pretrained=pretrained).to(self.device)
        self.resnet_flow.eval()
        for param in self.resnet_flow.parameters():
            param.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Input frames: (B, K, C, H, W) where K=6, C=3, H=256, W=344.
        Returns: (B, 2048) teacher feature representation.
        """
        B, K, C, H, W = frames.shape
        flow_pairs = []
        for t in range(K - 1):
            # Compute inter-frame displacement representation (2 channels per pair)
            diff = frames[:, t + 1] - frames[:, t]
            flow_2ch = diff[:, :2, :, :]
            flow_pairs.append(flow_2ch)
        
        flow_stack = torch.cat(flow_pairs, dim=1)  # (B, 10, H, W)
        return self.resnet_flow(flow_stack)


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

        if len(dataloader) == 0:
            return 0.0

        for batch_idx, batch_data in enumerate(dataloader):
            # Support both 'clip' and 'frames' dictionary keys from dataset loaders
            frames = batch_data.get("clip", batch_data.get("frames"))
            if frames is None:
                raise KeyError("Batch data must contain 'clip' or 'frames' tensor.")
            frames = frames.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass through frozen teacher
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


def run_kd_training(config: dict, args=None):
    """
    Main execution routine for Knowledge Distillation training stage.
    Trains IMFE student against the frozen teacher on the ActivityNet subset.
    """
    from data.activitynet import ActivityNetSubsetDataset

    device_str = (args.device if args and hasattr(args, "device") and args.device else config.get("training", {}).get("device", "cpu"))
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    logger.info(f"Knowledge Distillation running on device: {device}")

    # Dataset & DataLoader setup
    dataset_cfg = config.get("dataset", {})
    data_root = (args.data_root if args and hasattr(args, "data_root") and args.data_root else dataset_cfg.get("data_root", "data/activitynet/"))
    logger.info(f"Loading ActivityNet dataset from: {data_root}")

    train_split = dataset_cfg.get("train_split", "train")
    train_dataset = ActivityNetSubsetDataset(data_root=data_root, split=train_split)
    logger.info(f"ActivityNet dataset loaded: {len(train_dataset)} clips found.")

    if len(train_dataset) == 0:
        logger.error(f"No clips found in {data_root}. Run scripts/prepare_datasets.py first.")
        return

    batch_size = config.get("training", {}).get("batch_size", 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Models & Loss initialization
    clip_len = config.get("training", {}).get("clip_length", 6)
    student = IMFE(clip_length=clip_len, feature_dim=2048)
    teacher = TeacherFlowPipeline(in_channels=10, pretrained=False, device=device)
    
    # Support standard MSE loss or novel CosineMSELoss extension
    loss_type = config.get("distillation", {}).get("loss", "mse")
    if loss_type == "cosine_mse":
        try:
            from extensions.losses import CosineMSELoss
            criterion = CosineMSELoss()
        except ImportError:
            criterion = FeatureDistillationLoss()
    else:
        criterion = FeatureDistillationLoss()

    lr = float(config.get("training", {}).get("learning_rate", 1e-4))
    weight_decay = float(config.get("training", {}).get("weight_decay", 1e-5))
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

    trainer = KnowledgeDistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    epochs = config.get("training", {}).get("epochs", 5)
    checkpoint_dir = config.get("training", {}).get("checkpoint_dir", "checkpoints/")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Starting distillation training for {epochs} epochs...")
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        epoch_loss = trainer.train_epoch(train_loader)
        logger.info(f"Epoch [{epoch}/{epochs}] - KD Distillation Loss: {epoch_loss:.6f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_path = os.path.join(checkpoint_dir, "imfe_best.pt")
            torch.save(student.state_dict(), ckpt_path)
            logger.info(f"Saved best IMFE checkpoint to: {ckpt_path}")

    logger.info(f"[SUCCESS] Knowledge Distillation finished. Best Loss: {best_loss:.6f}")
