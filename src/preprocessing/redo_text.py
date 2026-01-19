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
    filename='redo_text.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def redo_text_preprocessing(annotations_csv, data_root, output_root):
    """
    Redoes ONLY text preprocessing (transcription + embeddings) with forced English.
    """
    config = PreprocessingConfig()
    df = pd.read_csv(annotations_csv)
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    text_preprocessor = TextPreprocessor(
        asr_model=config.asr_model,
        embedding_model=config.embedding_model
    )
    
    print("Redoing text preprocessing (Forced English)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        user_run_id = f"user_{row['usernum']}_run_{row['run']}"
        txt_path = output_root / "text" / f"{user_run_id}.txt"
        pt_path = output_root / "text" / f"{user_run_id}.pt"
        
        visual_path = data_root / row['video'].lstrip('./')
        audio_path = visual_path.parent / 'video.mp4'
        
        try:
            # 1. Force English Transcription
            text = text_preprocessor.transcribe(str(audio_path))
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
                
            # 2. New BERT Embeddings from English text
            text_feat = text_preprocessor.embed(text, max_length=config.max_text_len)
            torch.save(text_feat, pt_path)
            
            logging.info(f"Fixed text for {user_run_id}")
        except Exception as e:
            logging.error(f"Failed to fix {user_run_id}: {str(e)}")

    print("\nText redo complete. Running aggregator...")
    # Import and run aggregator
    from src.preprocessing.aggregate import generate_processed_metadata
    generate_processed_metadata(str(output_root), str(annotations_csv))

if __name__ == "__main__":
    ANNOTATIONS = "c:/Users/saridb/BagOfLies/data/BagOfLies/Annotations.csv"
    DATA_ROOT = "c:/Users/saridb/BagOfLies/data/BagOfLies"
    OUTPUT_ROOT = "c:/Users/saridb/BagOfLies/data/processed"
    redo_text_preprocessing(ANNOTATIONS, DATA_ROOT, OUTPUT_ROOT)
