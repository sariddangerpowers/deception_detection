import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from pathlib import Path

from src.config import PreprocessingConfig
from .video import extract_video_features
from .audio import extract_audio_features
from .text import TextPreprocessor

class BagOfLiesDataset(Dataset):
    """
    Multimodal Dataset for BagOfLies.
    Coordinates Video, Audio, and Text extraction.
    
    Prepares for inconsistencies by handling missing files or modalities.
    """
    def __init__(self, annotations_csv, data_root, config: PreprocessingConfig = None, transform=None, use_asr=False):
        """
        Args:
            annotations_csv (str): Path to Annotations.csv.
            data_root (str): Root directory for 'Finalised' data.
            config (PreprocessingConfig): Configuration object.
            use_asr (bool): Whether to transcribe audio on the fly.
        """
        self.df = pd.read_csv(annotations_csv)
        self.data_root = Path(data_root)
        self.use_asr = use_asr
        self.config = config if config else PreprocessingConfig()
        
        if use_asr:
            self.text_preprocessor = TextPreprocessor(
                asr_model=self.config.asr_model,
                embedding_model=self.config.embedding_model
            )
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Paths (Annotations use relative paths like ./Finalised/...)
        video_rel = row['video'].lstrip('./')
        video_path = self.data_root / video_rel
        audio_path = video_path.parent / 'video.mp4' # Audio is inside the mp4
        
        # 2. Extract Modalities
        try:
            video_features = extract_video_features(
                str(video_path),
                num_frames=self.config.num_frames,
                target_fps=self.config.target_fps,
                target_size=self.config.target_size
            )
        except Exception as e:
            # Fallback/Placeholder if video fails
            print(f"Warning: Failed to load video at {video_path}: {e}")
            video_features = torch.zeros((1, self.config.num_frames, self.config.target_size[0], self.config.target_size[1]))
            
        try:
            audio_features = extract_audio_features(
                str(audio_path),
                target_sr=self.config.sample_rate
            )
        except Exception as e:
            print(f"Warning: Failed to load audio at {audio_path}: {e}")
            audio_features = torch.zeros((1, 100, 39)) # Placeholder
            
        # 3. Text (Transcribe or Placeholder)
        # In a real "perfect" run, we'd cache these.
        if self.use_asr:
            text = self.text_preprocessor.transcribe(str(audio_path))
            text_features = self.text_preprocessor.embed(text, max_length=self.config.max_text_len)
        else:
            # Placeholder text features (e.g., 300D or 768D)
            text_features = torch.zeros((1, self.config.max_text_len, 300))
            
        # 4. Label
        label = torch.tensor(row['truth'], dtype=torch.long)
        
        return {
            'video': video_features.squeeze(0), # (10, 224, 224)
            'audio': audio_features.squeeze(0), # (Seq, 39)
            'text': text_features.squeeze(0),   # (Seq, 300)
            'label': label
        }

if __name__ == "__main__":
    # Test script would go here
    pass
