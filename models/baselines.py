import torch
import torch.nn as nn
import torchvision.models as models

class RGBOnlyBaseline(nn.Module):
    """
    RGB-Only baseline using standalone ResNet-50 spatial features.
    Extracts 2048-dimensional spatial feature representations from RGB frames.
    """
    def __init__(self, pretrained: bool = True):
        super(RGBOnlyBaseline, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        # Remove the final classification layer to yield 2048-dim features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: (B, K, 3, H, W), (B, 3, H, W), or (K, 3, H, W)
        Returns feature tensor of shape (..., 2048)
        """
        is_5d = x.dim() == 5
        if is_5d:
            B, K, C, H, W = x.shape
            x = x.view(B * K, C, H, W)
            
        features = self.backbone(x) # Shape: (B*K, 2048, 1, 1)
        features = torch.flatten(features, 1) # Shape: (B*K, 2048)
        
        if is_5d:
            features = features.view(B, K, 2048)
            
        return features


class RAFTFlowExtractor:
    """
    Wrapper for pre-trained RAFT-Large from TorchVision for optical flow extraction.
    """
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = models.optical_flow.Raft_Large_Weights.DEFAULT
        self.raft_model = models.optical_flow.raft_large(weights=weights, progress=False).to(self.device)
        self.raft_model.eval()
        
    @torch.no_grad()
    def extract_flow(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        Extracts optical flow between two frames using RAFT-Large.
        Inputs image1, image2: Tensors of shape (B, 3, H, W) scaled to [0, 255].
        Returns optical flow tensor of shape (B, 2, H, W).
        """
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        
        # RAFT returns a list of flow predictions across refinement iterations; take the final prediction
        list_of_flows = self.raft_model(image1, image2)
        flow = list_of_flows[-1] # Shape: (B, 2, H, W)
        return flow