import torch
import pytest
from src.models.video_branch import VideoBranch
from src.models.sequence_branch import SequenceBranch
from src.models.fusion_model import build_proposed_model

def test_video_branch_output_shape():
    batch_size = 4
    num_frames = 10
    h, w = 224, 224
    model = VideoBranch(num_frames=num_frames, output_dim=128)
    
    # Input: (Batch, 1, Depth, H, W)
    x = torch.randn(batch_size, 1, num_frames, h, w)
    output = model(x)
    
    assert output.shape == (batch_size, 128)

def test_sequence_branch_audio_shape():
    batch_size = 8
    seq_len = 50
    input_dim = 39 # MFCCs
    model = SequenceBranch(input_dim=input_dim, output_dim=128)
    
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    
    assert output.shape == (batch_size, 128)

def test_sequence_branch_text_shape():
    batch_size = 2
    seq_len = 200
    input_dim = 300 # GloVe dim
    model = SequenceBranch(input_dim=input_dim, output_dim=128)
    
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    
    assert output.shape == (batch_size, 128)

def test_fusion_model_integration():
    model = build_proposed_model()
    batch_size = 2
    
    v = torch.randn(batch_size, 1, 10, 224, 224)
    a = torch.randn(batch_size, 100, 39)
    t = torch.randn(batch_size, 200, 300)
    
    output = model(v, a, t)
    assert output.shape == (batch_size, 2)

def test_fusion_model_missing_optional():
    from src.models.fusion_model import MultimodalFusionModel
    from src.models.video_branch import VideoBranch
    from src.models.audio_branch import AudioBranch
    from src.models.text_branch import TextBranch
    from src.models.dummy_branches import EEGBranch
    
    v_b = VideoBranch()
    a_b = AudioBranch()
    t_b = TextBranch()
    e_b = EEGBranch() # Placeholder
    
    model = MultimodalFusionModel(v_b, a_b, t_b, eeg_branch=e_b)
    
    batch_size = 3
    v = torch.randn(batch_size, 1, 10, 224, 224)
    a = torch.randn(batch_size, 50, 39)
    t = torch.randn(batch_size, 100, 300)
    
    # Run without EEG input, should use dummy zeros
    output = model(v, a, t, eeg_in=None)
    assert output.shape == (batch_size, 2)
