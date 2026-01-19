import os
import pandas as pd
import torch
import logging
import traceback
from tqdm import tqdm
from pathlib import Path

from src.config import PreprocessingConfig
from src.preprocessing.video import extract_video_features
from src.preprocessing.audio import extract_audio_features
from src.preprocessing.text import TextPreprocessor

# Configure Logging
logging.basicConfig(
    filename='preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_batch_preprocessing(annotations_csv, data_root, output_root):
    """
    Processes all samples in the BagOfLies dataset and saves features.
    """
    config = PreprocessingConfig()
    df = pd.read_csv(annotations_csv)
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    # Initialize Text Preprocessor
    text_preprocessor = TextPreprocessor(
        asr_model=config.asr_model,
        embedding_model=config.embedding_model
    )
    
    # Ensure output directories exist
    (output_root / "video").mkdir(parents=True, exist_ok=True)
    (output_root / "audio").mkdir(parents=True, exist_ok=True)
    (output_root / "text").mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total": len(df),
        "success": 0,
        "failed": 0,
        "errors": []
    }
    
    print(f"Starting preprocessing for {len(df)} samples...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            user_run_id = f"user_{row['usernum']}_run_{row['run']}"
            
            # Paths
            video_rel = row['video'].lstrip('./')
            video_path = data_root / video_rel
            audio_path = video_path.parent / 'video.mp4' # Audio is in the mp4
            
            # 1. Process Video
            video_feat = extract_video_features(
                str(video_path),
                num_frames=config.num_frames,
                target_fps=config.target_fps,
                target_size=config.target_size
            )
            torch.save(video_feat, output_root / "video" / f"{user_run_id}.pt")
            
            # 2. Process Audio
            audio_feat = extract_audio_features(
                str(audio_path),
                target_sr=config.sample_rate
            )
            torch.save(audio_feat, output_root / "audio" / f"{user_run_id}.pt")
            
            # 3. Process Text
            # First transcribe
            text = text_preprocessor.transcribe(str(audio_path))
            # Save raw text for human review
            with open(output_root / "text" / f"{user_run_id}.txt", "w", encoding="utf-8") as f:
                f.write(text)
            # Then embed
            text_feat = text_preprocessor.embed(text, max_length=config.max_text_len)
            torch.save(text_feat, output_root / "text" / f"{user_run_id}.pt")
            
            stats["success"] += 1
            logging.info(f"Successfully processed {user_run_id}")
            
        except Exception as e:
            stats["failed"] += 1
            tb_str = traceback.format_exc()
            error_msg = f"Error processing sample {idx} ({user_run_id if 'user_run_id' in locals() else 'unknown'}):\n{tb_str}"
            stats["errors"].append(error_msg)
            logging.error(error_msg)
            
    # Save final stats
    with open(output_root / "preprocessing_report.txt", "w") as f:
        f.write(f"Preprocessing Report\n")
        f.write(f"====================\n")
        f.write(f"Total Samples: {stats['total']}\n")
        f.write(f"Successful: {stats['success']}\n")
        f.write(f"Failed: {stats['failed']}\n\n")
        if stats["errors"]:
            f.write("Errors:\n")
            for err in stats["errors"]:
                f.write(f"- {err}\n")
                
    print(f"\nPreprocessing Complete!")
    print(f"Success: {stats['success']}, Failed: {stats['failed']}")
    print(f"Report saved to {output_root / 'preprocessing_report.txt'}")

if __name__ == "__main__":
    ANNOTATIONS = "c:/Users/saridb/BagOfLies/data/BagOfLies/Annotations.csv"
    DATA_ROOT = "c:/Users/saridb/BagOfLies/data/BagOfLies"
    OUTPUT_ROOT = "c:/Users/saridb/BagOfLies/data/processed"
    
    run_batch_preprocessing(ANNOTATIONS, DATA_ROOT, OUTPUT_ROOT)
