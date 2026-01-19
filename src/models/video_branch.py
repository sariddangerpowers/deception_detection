import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig

class VideoBranch(nn.Module):
    """
    Video Branch using 3D-CNN to extract spatial-temporal features from video frames.
    
    Architecture based on Sehrawat et al. (2023):
    - 3D-CNN layers for spatial-temporal feature extraction.
    - Global Average Pooling for dimensionality reduction.
    - Dense layers for final modality features.
    """
    def __init__(self, config: ModelConfig = None):
        super(VideoBranch, self).__init__()
        
        if config is None:
            config = ModelConfig()
            
        # input shape: (Batch, 1, num_frames, H, W) - Grayscale
        self.conv_stack = nn.Sequential(
            # Layer 1
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            # Layer 3
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            # Layer 5
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding='same'),
            nn.ReLU(),
            # In the paper they Pool here, then GlobalAvgPool
            # We'll use AdaptiveAvgPool3d to be robust to varying dimensions
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_video),
            nn.Linear(256, config.video_output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (Batch, 1, num_frames, H, W)
        Returns:
            Tensor: Feature vector of shape (Batch, output_dim)
        """
        # Ensure input is 5D: (Batch, Channels, Depth/Time, Height, Width)
        if x.dim() == 4:
            x = x.unsqueeze(1)
            
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x

if __name__ == "__main__":
    # Quick shape test
    model = VideoBranch()
    dummy_input = torch.randn(8, 1, 10, 224, 224) 
    output = model(dummy_input)
    print(f"Video output shape: {output.shape}") # Expected: [8, 128]
