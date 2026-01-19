import torch
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.preprocessing.video import extract_video_features
from src.preprocessing.audio import extract_audio_features
from src.config import PreprocessingConfig

@patch('cv2.VideoCapture')
def test_extract_video_features_mock(mock_video):
    # Setup mock
    instance = mock_video.return_value
    instance.isOpened.return_value = True
    instance.get.side_effect = [30.0, 100] # FPS, Frame Count
    instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    
    num_frames = 5
    config = PreprocessingConfig(num_frames=num_frames, target_size=(224, 224))
    
    output = extract_video_features("dummy.mp4", num_frames=num_frames, target_size=config.target_size)
    
    # Output should be (1, num_frames, H, W)
    assert output.shape == (1, 5, 224, 224)
    assert output.max() <= 1.0
    assert output.min() >= 0.0

@patch('librosa.load')
@patch('librosa.feature.mfcc')
@patch('librosa.feature.delta')
def test_extract_audio_features_mock(mock_delta, mock_mfcc, mock_load):
    # Setup mocks
    mock_load.return_value = (np.random.randn(16000), 16000)
    # mock_mfcc returns (13, seq_len)
    mock_mfcc.return_value = np.zeros((13, 100))
    mock_delta.side_effect = [np.zeros((13, 100)), np.zeros((13, 100))] # delta, delta2
    
    output = extract_audio_features("dummy.wav", target_sr=16000)
    
    # Output should be (1, Sequence, 39)
    assert output.shape == (1, 100, 39)
    assert torch.is_tensor(output)

def test_preprocessing_config_defaults():
    config = PreprocessingConfig()
    assert config.num_frames == 10
    assert config.target_size == (224, 224)
    assert config.sample_rate == 16000
