import os
import pandas as pd
import torch
import logging
from pathlib import Path
from tqdm import tqdm

from src.config import PreprocessingConfig
from src.preprocessing.text import TextPreprocessor

# Configure Logging
logging.basicConfig(
    filename='fix_failed.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fix_failed_samples(annotations_csv, data_root, output_root):
    """
    Identifies samples that lack text features and re-processes them.
    """
    config = PreprocessingConfig()
    df = pd.read_csv(annotations_csv)
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    text_preprocessor = TextPreprocessor(
        asr_model=config.asr_model,
        embedding_model=config.embedding_model
    )
    
    success_count = 0
    failed_count = 0
    
    print("Checking for failed samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        user_run_id = f"user_{row['usernum']}_run_{row['run']}"
        text_feat_path = output_root / "text" / f"{user_run_id}.pt"
        
        if not text_feat_path.exists():
            # Found a missing/failed one
            visual_path = data_root / row['video'].lstrip('./')
            audio_path = visual_path.parent / 'video.mp4'
            
            try:
                print(f"\nProcessing failed sample: {user_run_id}")
                # Transcribe with chunking (enabled in updated TextPreprocessor)
                text = text_preprocessor.transcribe(str(audio_path))
                # Embed
                text_feat = text_preprocessor.embed(text, max_length=config.max_text_len)
                torch.save(text_feat, text_feat_path)
                
                success_count += 1
                logging.info(f"Successfully fixed {user_run_id}")
            except Exception as e:
                failed_count += 1
                logging.error(f"Still failed {user_run_id}: {str(e)}")
                
    print(f"\nFixing Complete! Newly success: {success_count}, Still failed: {failed_count}")

if __name__ == "__main__":
    ANNOTATIONS = "c:/Users/saridb/BagOfLies/data/BagOfLies/Annotations.csv"
    DATA_ROOT = "c:/Users/saridb/BagOfLies/data/BagOfLies"
    OUTPUT_ROOT = "c:/Users/saridb/BagOfLies/data/processed"
    
    fix_failed_samples(ANNOTATIONS, DATA_ROOT, OUTPUT_ROOT)
