import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.config import PreprocessingConfig
from src.preprocessing.text import TextPreprocessor

def extract_all_transcripts(annotations_csv, data_root, output_root):
    """
    Focused script to extract ONLY raw text transcriptions for all samples.
    """
    config = PreprocessingConfig()
    df = pd.read_csv(annotations_csv)
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    text_preprocessor = TextPreprocessor(
        asr_model=config.asr_model,
        embedding_model=config.embedding_model
    )
    
    (output_root / "text").mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting transcriptions for {len(df)} samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        user_run_id = f"user_{row['usernum']}_run_{row['run']}"
        txt_path = output_root / "text" / f"{user_run_id}.txt"
        
        # We skip if already exists (paranoia)
        if not txt_path.exists():
            visual_path = data_root / row['video'].lstrip('./')
            audio_path = visual_path.parent / 'video.mp4'
            
            try:
                # Transcribe
                text = text_preprocessor.transcribe(str(audio_path))
                # Save TXT
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception as e:
                print(f"\nFailed {user_run_id}: {e}")

if __name__ == "__main__":
    ANNOTATIONS = "c:/Users/saridb/BagOfLies/data/BagOfLies/Annotations.csv"
    DATA_ROOT = "c:/Users/saridb/BagOfLies/data/BagOfLies"
    OUTPUT_ROOT = "c:/Users/saridb/BagOfLies/data/processed"
    extract_all_transcripts(ANNOTATIONS, DATA_ROOT, OUTPUT_ROOT)
