from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class PreprocessingConfig:
    # Video
    num_frames: int = 10
    target_fps: int = 10
    target_size: Tuple[int, int] = (224, 224)
    
    # Audio
    sample_rate: int = 16000
    n_mfcc: int = 13
    
    # Text
    max_text_len: int = 200
    asr_model: str = "openai/whisper-tiny"
    embedding_model: str = "bert-base-uncased"

@dataclass
class ModelConfig:
    # Modal dimensions
    video_output_dim: int = 128
    audio_output_dim: int = 128
    text_output_dim: int = 128
    
    # Input dimensions (Modal specifics)
    text_input_dim: int = 768 # BERT-base-uncased
    audio_input_dim: int = 39  # 13 MFCC + Delta + Delta-Delta
    
    # Sequence settings
    audio_hidden_dim: int = 128
    text_hidden_dim: int = 128
    
    # Regularization
    dropout_video: float = 0.3
    dropout_audio: float = 0.3
    dropout_text: float = 0.3
    dropout_fusion: float = 0.5
    
    # Fusion head
    dense_layers: List[int] = field(default_factory=lambda: [512, 256, 128])

@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 15
    data_split: float = 0.7 # 70% train
