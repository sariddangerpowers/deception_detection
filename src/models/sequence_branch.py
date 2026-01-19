import torch
import torch.nn as nn

from src.config import ModelConfig

class SequenceBranch(nn.Module):
    """
    Modular sequence branch using Stacked Bi-LSTMs.
    Shared logic for Audio (MFCC) and Text (Embeddings) modalities.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=128, dropout=0.3, config: ModelConfig = None):
        super(SequenceBranch, self).__init__()
        
        # If config is provided, we can override default params if needed
        # For now, we use explicitly passed params or config ones
        if config:
            dropout = config.dropout_audio # simplified assumption for shared logic
            output_dim = config.audio_output_dim
            
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Bi-directional output is twice the hidden_dim
        lstm_out_dim = hidden_dim * 2
        
        self.fc_stack = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (Batch, SeqLen, input_dim)
        Returns:
            Tensor: Feature vector of shape (Batch, output_dim)
        """
        # lstm_out: (Batch, SeqLen, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # Take mean across sequence dimension (Global Average Pooling)
        # Paper says "GlobalAvgPool" after Bi-LSTM
        pooled = torch.mean(lstm_out, dim=1)
        
        x = self.fc_stack(pooled)
        return x

if __name__ == "__main__":
    # Quick test
    # Audio test: 39 MFCC features
    audio_model = SequenceBranch(input_dim=39)
    dummy_audio = torch.randn(8, 100, 39)
    audio_out = audio_model(dummy_audio)
    print(f"Audio output shape: {audio_out.shape}") # Expect [8, 128]
    
    # Text test: 300D embeddings
    text_model = SequenceBranch(input_dim=300)
    dummy_text = torch.randn(8, 200, 300)
    text_out = text_model(dummy_text)
    print(f"Text output shape: {text_out.shape}") # Expect [8, 128]
