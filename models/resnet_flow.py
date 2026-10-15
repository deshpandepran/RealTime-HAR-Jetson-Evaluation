import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFlowExtractor(nn.Module):
    """
    ResNet-50 Flow Feature Extractor module.
    Processes stacked optical flow frames into 2048-dimensional feature representations.
    """
    def __init__(self, in_channels: int = 10, pretrained: bool = True):
        super(ResNetFlowExtractor, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Adapt the first convolution layer to accept multi-channel flow inputs (default: 10 channels for 5 flow frames)
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        
        if pretrained:
            with torch.no_grad():
                w = original_conv1.weight.data # (64, 3, 7, 7)
                # Replicate or adapt RGB weights across the new input channel dimension
                rep_w = w.repeat(1, (in_channels // 3) + 1, 1, 1)[:, :in_channels, :, :]
                rep_w = rep_w * (3.0 / in_channels)
                self.conv1.weight.copy_(rep_w)
                
        # Retain subsequent layers from ResNet-50 backbone
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
    def forward(self, flow_stack: torch.Tensor) -> torch.Tensor:
        """
        Input flow_stack shape: (B, in_channels, H, W) (e.g., in_channels = 10 for 5 optical flow frames).
        Returns feature tensor of shape (B, 2048).
        """
        x = self.conv1(flow_stack)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Shape: (B, 2048)
        
        return x