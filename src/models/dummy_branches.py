import torch
import torch.nn as nn

class DummyBranch(nn.Module):
    """
    Placeholder branch for future modalities (e.g., EEG, Gaze).
    Produces a zero feature vector to maintain uncoupling and architectural integrity.
    """
    def __init__(self, output_dim=128):
        super(DummyBranch, self).__init__()
        self.output_dim = output_dim
        # Minimal parameter to avoid optimizer errors if we were to train
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x=None):
        """
        Args:
            x: Input (ignored)
        Returns:
            Tensor: Zero feature vector of shape (Batch, output_dim)
        """
        # We need a batch size. If x is provided, use its first dimension.
        # Otherwise, assume batch size 1 for dummy testing.
        batch_size = x.shape[0] if x is not None else 1
        device = self.dummy_param.device
        return torch.zeros(batch_size, self.output_dim, device=device)

class EEGBranch(DummyBranch):
    """Placeholder for EEG processing."""
    pass

class GazeBranch(DummyBranch):
    """Placeholder for Gaze processing."""
    pass
