from dataclasses import dataclass, field
from typing import Tuple, List
from pathlib import Path
import os

@dataclass
class PathConfig:
    # Project Root: Default to the parent of the 'src' directory
    project_root: Path = Path(__file__).resolve().parent.parent
    
    # Data Directories (Relative to project root by default)
    data_root: Path = project_root / "data"
    bag_of_lies_dir: Path = data_root / "BagOfLies"
    processed_dir: Path = data_root / "processed"
    
    # Files
    annotations_csv: Path = bag_of_lies_dir / "Annotations.csv"
    metadata_csv: Path = processed_dir / "metadata.csv"
    
    def __post_init__(self):
        # Allow environment variable overrides for Colab/Server usage
        if "PROJECT_ROOT" in os.environ:
            self.project_root = Path(os.environ["PROJECT_ROOT"])
        if "DATA_ROOT" in os.environ:
            self.data_root = Path(os.environ["DATA_ROOT"])
            self.bag_of_lies_dir = self.data_root / "BagOfLies"
            self.processed_dir = self.data_root / "processed"
            self.annotations_csv = self.bag_of_lies_dir / "Annotations.csv"
            self.metadata_csv = self.processed_dir / "metadata.csv"

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
