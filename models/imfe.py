import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Standard ResNet Residual Block with two 3x3 convolutions and shortcut connection.
    """
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class BatchedFeatureEncoder(nn.Module):
    """
    Encodes a batch of K consecutive video frames into feature maps at 1/8 scale.
    Input: (B*K, 3, H, W)
    Output: (B*K, D, H/8, W/8) where D=256, H/8=32, W/8=43 for 256x344 input.
    """
    def __init__(self, out_channels: int = 256, num_blocks: int = 6):
        super(BatchedFeatureEncoder, self).__init__()
        # Stem: downsample by 8 using strided convolutions
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), # H/2, W/2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # H/4, W/4
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1, bias=False), # H/8, W/8
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 6 residual blocks
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return x


class CorrelationModule(nn.Module):
    """
    Computes feature-wise dot-product correlations between consecutive frames.
    Input feature maps: (B, K, D, H', W')
    Output: K-1 correlation volumes concatenated: (B, (K-1) * (H'*W'), H', W').
    """
    def __init__(self):
        super(CorrelationModule, self).__init__()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, K, D, H_prime, W_prime = features.shape
        N = H_prime * W_prime
        
        correlations = []
        for t in range(K - 1):
            f1 = features[:, t, :, :, :]     # (B, D, H', W')
            f2 = features[:, t+1, :, :, :]   # (B, D, H', W')
            
            # Flatten spatial dimensions of f1 to (B, D, N)
            f1_flat = f1.view(B, D, N)
            
            # Compute dot product correlation between f2 and f1_flat -> (B, N, H', W')
            corr = torch.einsum('bdhw,bdn->bnhw', f2, f1_flat)
            correlations.append(corr)
            
        # Concatenate along channel dimension: (B, (K-1)*N, H', W')
        return torch.cat(correlations, dim=1)


class CompressionAndEncoding(nn.Module):
    """
    Compression & Encoding Module containing 5 Conv2D layers to compress data dimensionality
    from concatenated correlation volumes to 512 channels.
    """
    def __init__(self, in_channels: int, out_channels: int = 512):
        super(CompressionAndEncoding, self).__init__()
        # 5 Conv2D layers
        self.conv1 = nn.Conv2d(in_channels, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)
        
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        return x


class FinalFeatureEncoder(nn.Module):
    """
    Final Feature Encoder containing 9 residual blocks followed by global pooling and projection to 2048-dim.
    """
    def __init__(self, in_channels: int = 512, num_blocks: int = 9, feature_dim: int = 2048):
        super(FinalFeatureEncoder, self).__init__()
        self.transition = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        
        self.blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(num_blocks)])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.transition(x)))
        x = self.blocks(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class IMFE(nn.Module):
    """
    Single-Shot Integrated Motion Feature Extractor (IMFE) architecture.
    Replaces traditional optical flow extraction + Flow Feature Extractor in a single pass.
    """
    def __init__(self, clip_length: int = 6, feature_dim: int = 2048):
        super(IMFE, self).__init__()
        self.clip_length = clip_length
        
        # 1. Batched Feature Encoder (6 residual blocks, D=256)
        self.batched_encoder = BatchedFeatureEncoder(out_channels=256, num_blocks=6)
        
        # 2. Correlation Module
        self.correlation_module = CorrelationModule()
        
        # For K=6, K-1=5 correlation volumes, each with N = 32 * 43 = 1376 channels.
        # Total concatenated channels = 5 * 1376 = 6880
        concat_channels = (clip_length - 1) * (32 * 43)
        
        # 3. Compression & Encoding Module (5 Conv2D layers)
        self.compression_module = CompressionAndEncoding(in_channels=concat_channels, out_channels=512)
        
        # 4. Final Feature Encoder (9 residual blocks -> 2048-dim feature)
        self.final_encoder = FinalFeatureEncoder(in_channels=512, num_blocks=9, feature_dim=feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: (B, K, 3, H, W) where K=6, H=256, W=344
        Returns motion feature tensor of shape (B, 2048)
        """
        B, K, C, H, W = x.shape
        assert K == self.clip_length, f"Expected clip length {self.clip_length}, got {K}"
        
        # Reshape to (B * K, C, H, W) for batched feature encoding
        x_flat = x.view(B * K, C, H, W)
        encoded_features = self.batched_encoder(x_flat) # (B*K, D, H/8, W/8)
        
        # Reshape back to (B, K, D, H/8, W/8)
        _, D, H_prime, W_prime = encoded_features.shape
        encoded_features = encoded_features.view(B, K, D, H_prime, W_prime)
        
        # Compute correlation volumes and concatenate: (B, (K-1)*N, H', W')
        corr_concat = self.correlation_module(encoded_features)
        
        # Compression & Encoding
        compressed = self.compression_module(corr_concat) # (B, 512, H', W')
        
        # Final Feature Encoding -> (B, 2048)
        motion_features = self.final_encoder(compressed)
        
        return motion_features