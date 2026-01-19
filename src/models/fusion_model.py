import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig

class MultimodalFusionModel(nn.Module):
    """
    Multimodal Fusion Model that integrates features from Video, Audio, and Text branches.
    Designed to be modular and uncoupled, allowing easy addition or removal of branches.
    """
    def __init__(self, video_branch, audio_branch, text_branch, eeg_branch=None, gaze_branch=None, config: ModelConfig = None):
        super(MultimodalFusionModel, self).__init__()
        
        if config is None:
            config = ModelConfig()
            
        self.config = config
        self.video_branch = video_branch
        self.audio_branch = audio_branch
        self.text_branch = text_branch
        self.eeg_branch = eeg_branch
        self.gaze_branch = gaze_branch
        
        # Calculate total fusion dimension
        # Each active branch produces a feature vector of length 128 by default
        total_dim = 128 * 3 # Video, Audio, Text
        if self.eeg_branch:
            total_dim += 128
        if self.gaze_branch:
            total_dim += 128
            
        # Classification Head
        self.classifier = nn.Sequential(
            # Dense(512)
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            
            # Dense(256)
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Dense(128)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Dense(2) - Softmax is handled by CrossEntropyLoss or explicitly
            nn.Linear(128, 2)
        )

    def forward(self, video_in, audio_in, text_in, eeg_in=None, gaze_in=None):
        """
        Multimodal forward pass.
        Handles cases where inputs might be missing by using placeholder logic or defaults.
        """
        # Extract features from each modality
        v_feat = self.video_branch(video_in)
        a_feat = self.audio_branch(audio_in)
        t_feat = self.text_branch(text_in)
        
        features = [v_feat, a_feat, t_feat]
        
        if self.eeg_branch and eeg_in is not None:
            features.append(self.eeg_branch(eeg_in))
        elif self.eeg_branch:
            # Prepare for inconsistencies: create dummy features if branch exists but input is missing
            features.append(self.eeg_branch(None).expand(v_feat.shape[0], -1))

        if self.gaze_branch and gaze_in is not None:
            features.append(self.gaze_branch(gaze_in))
        elif self.gaze_branch:
            features.append(self.gaze_branch(None).expand(v_feat.shape[0], -1))
            
        # Fusion via concatenation
        fused = torch.cat(features, dim=1)
        
        # Final classification
        logits = self.classifier(fused)
        return logits

def build_multimodal_model(config=None):
    """Factory function to build the architecture proposed in the paper."""
    from .video_branch import VideoBranch
    from .audio_branch import AudioBranch
    from .text_branch import TextBranch
    
    video = VideoBranch()
    audio = AudioBranch()
    text = TextBranch()
    
    return MultimodalFusionModel(video, audio, text, config=config)

if __name__ == "__main__":
    # Test the integrated model
    model = build_proposed_model()
    
    # Dummy inputs for one batch of size 4
    v = torch.randn(4, 1, 10, 224, 224)
    a = torch.randn(4, 100, 39)
    t = torch.randn(4, 200, 300)
    
    out = model(v, a, t)
    print(f"Fusion model output shape: {out.shape}") # Expect [4, 2]
    print(f"Probabilities (softmax): {F.softmax(out, dim=1)}")
