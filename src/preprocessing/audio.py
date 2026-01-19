import librosa
import numpy as np
import torch

def extract_audio_features(audio_path, target_sr=16000):
    """
    Extracts and preprocesses MFCC features from an audio file.
    
    Preprocessing steps (Sehrawat et al. 2023):
    1. Resample to 16kHz.
    2. Extract 13 MFCC coefficients.
    3. Compute 1st order (Delta) and 2nd order (Delta-Delta) derivatives.
    4. Z-score normalization.
    
    Args:
        audio_path (str): Path to the audio file.
        target_sr (int): Sampling rate for processing.
        
    Returns:
        torch.Tensor: Feature matrix of shape (Batch=1, Sequence, 39)
    """
    y, sr = librosa.load(audio_path, sr=target_sr)
    
    # 1. Extract 13 MFCCs
    # Using parameters mentioned in instruction MD: 25ms window, 10ms hop
    n_fft = int(target_sr * 0.025) # 400
    hop_length = int(target_sr * 0.010) # 160
    
    mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    
    # 2. Compute Deltas and Delta-Deltas
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # 3. Concatenate all features (39 total)
    # Shape: (39, sequence_length) -> Transpose to (sequence_length, 39)
    combined = np.concatenate([mfcc, delta, delta2], axis=0).T
    
    # 4. Z-score normalization (Per-utterance)
    mean = np.mean(combined, axis=0)
    std = np.std(combined, axis=0) + 1e-8
    normalized = (combined - mean) / std
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
    
    return tensor

if __name__ == "__main__":
    # Example usage
    # out = extract_audio_features("path/to/audio.wav")
    # print(f"Audio feature shape: {out.shape}")
    pass
