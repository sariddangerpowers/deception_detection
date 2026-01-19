import os
import pandas as pd
from pathlib import Path
from src.config import PathConfig

def generate_processed_metadata(processed_root, original_annotations):
    """
    Scans the processed directories and creates a new metadata file
    linking results to labels.
    """
    processed_root = Path(processed_root)
    original_df = pd.read_csv(original_annotations)
    
    video_dir = processed_root / "video"
    audio_dir = processed_root / "audio"
    text_dir = processed_root / "text"
    
    rows = []
    
    for idx, row in original_df.iterrows():
        user_run_id = f"user_{row['usernum']}_run_{row['run']}"
        
        v_path = video_dir / f"{user_run_id}.pt"
        a_path = audio_dir / f"{user_run_id}.pt"
        t_path = text_dir / f"{user_run_id}.pt"
        txt_path = text_dir / f"{user_run_id}.txt"
        
        # We only include samples where all 3 modalities were successfully processed
        if v_path.exists() and a_path.exists() and t_path.exists():
            # Read transcription if it exists
            transcription = ""
            if txt_path.exists():
                with open(txt_path, "r", encoding="utf-8") as f:
                    transcription = f.read()
            
            rows.append({
                "sample_id": user_run_id,
                "usernum": row['usernum'],
                "video_feat": (v_path.relative_to(processed_root)).as_posix(),
                "audio_feat": (a_path.relative_to(processed_root)).as_posix(),
                "text_feat": (t_path.relative_to(processed_root)).as_posix(),
                "transcription": transcription,
                "label": row['truth']
            })
            
    meta_df = pd.DataFrame(rows)
    output_path = processed_root / "metadata.csv"
    meta_df.to_csv(output_path, index=False)
    print(f"Generated metadata for {len(rows)} samples at {output_path}")

if __name__ == "__main__":
    path_cfg = PathConfig()
    generate_processed_metadata(
        path_cfg.processed_dir,
        path_cfg.annotations_csv
    )
