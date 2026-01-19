import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from src.config import PathConfig

class PreprocessedBagOfLiesDataset(Dataset):
    """
    Dataset that loads pre-extracted features from disk.
    """
    def __init__(self, metadata_csv):
        self.df = pd.read_csv(metadata_csv)
        self.processed_root = Path(metadata_csv).parent
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load tensors (resolve relative paths stored in CSV)
        video_feat = torch.load(self.processed_root / row['video_feat']) 
        audio_feat = torch.load(self.processed_root / row['audio_feat']) 
        text_feat = torch.load(self.processed_root / row['text_feat'])   
        
        # Squeeze batch dimensions if they exist from saving
        if video_feat.dim() == 2: video_feat = video_feat.squeeze(0)
        if audio_feat.dim() == 2: audio_feat = audio_feat.squeeze(0)
        if text_feat.dim() == 3: text_feat = text_feat.squeeze(0)
        
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return {
            'video': video_feat,
            'audio': audio_feat,
            'text': text_feat,
            'label': label
        }

def multimodal_collate(batch):
    """
    Custom collate to pad variable length audio and text features.
    """
    video = torch.stack([item['video'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Pad Audio: (batch, seq, 39)
    audio_list = [item['audio'] for item in batch]
    # Check if they are already squeezed or have extra dim
    if audio_list[0].dim() == 3: # (1, seq, 39)
        audio_list = [a.squeeze(0) for a in audio_list]
    audio_padded = torch.nn.utils.rnn.pad_sequence(audio_list, batch_first=True)
    
    # Pad Text: (batch, seq, 768)
    text_list = [item['text'] for item in batch]
    if text_list[0].dim() == 3: # (1, seq, 768)
        text_list = [t.squeeze(0) for t in text_list]
    text_padded = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    
    return {
        'video': video,
        'audio': audio_padded,
        'text': text_padded,
        'label': labels
    }

def get_user_independent_splits(metadata_csv, test_size=0.3, seed=42):
    """
    Splits users into 70% Train/Val and 30% Test.
    Returns the dataframe and the indices for the split.
    """
    df = pd.read_csv(metadata_csv)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    
    # groups are 'usernum'
    groups = df['usernum']
    train_val_idx, test_idx = next(gss.split(df, groups=groups))
    
    return df, train_val_idx, test_idx

def get_cv_loaders(metadata_csv, train_val_idx, n_splits=5, batch_size=32, seed=42):
    """
    Generator for 5-Fold Cross Validation loaders on the training set.
    Ensures users are disconnected between folds.
    """
    df = pd.read_csv(metadata_csv)
    dataset = PreprocessedBagOfLiesDataset(metadata_csv)
    
    # Sub-select the train_val portion of the dataframe
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    groups = train_val_df['usernum']
    
    gkf = GroupKFold(n_splits=n_splits)
    
    for fold, (train_fold_idx, val_fold_idx) in enumerate(gkf.split(train_val_df, groups=groups)):
        # Map back to original dataset indices
        real_train_idx = train_val_idx[train_fold_idx]
        real_val_idx = train_val_idx[val_fold_idx]
        
        train_sub = Subset(dataset, real_train_idx)
        val_sub = Subset(dataset, real_val_idx)
        
        train_loader = DataLoader(
            train_sub, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=multimodal_collate,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_sub, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=multimodal_collate,
            num_workers=4,
            pin_memory=True
        )
        
        yield fold, train_loader, val_loader

def get_final_loaders(metadata_csv, train_val_idx, test_idx, batch_size=32):
    """
    Creates final loaders for retraining on full Train+Val and testing on Test.
    """
    dataset = PreprocessedBagOfLiesDataset(metadata_csv)
    
    train_set = Subset(dataset, train_val_idx)
    test_set = Subset(dataset, test_idx)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=multimodal_collate,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=multimodal_collate,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    path_cfg = PathConfig()
    META = str(path_cfg.metadata_csv)
    if Path(META).exists():
        df, train_val_idx, test_idx = get_user_independent_splits(META)
        
        # Verify user isolation
        train_users = set(df.iloc[train_val_idx]['usernum'])
        test_users = set(df.iloc[test_idx]['usernum'])
        overlap = train_users.intersection(test_users)
        
        print(f"Total Samples: {len(df)}")
        print(f"Train/Val Users: {len(train_users)}, Test Users: {len(test_users)}")
        print(f"Overlap: {overlap}")
        print(f"Split sizes: Train/Val={len(train_val_idx)}, Test={len(test_idx)}")
        
        # Test Fold generation
        for fold, tl, vl in get_cv_loaders(META, train_val_idx):
            print(f"Fold {fold}: Train={len(tl.dataset)}, Val={len(vl.dataset)}")
            break # Just one
