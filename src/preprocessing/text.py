import torch
import librosa
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModel
import numpy as np

class TextPreprocessor:
    """
    Handles transcription of audio and conversion of text to embeddings.
    
    Refinement: Uses modern transformer models for both ASR and Embeddings,
    as exact reproduction of 2023 GloVe setup depends on external static files.
    """
    def __init__(self, asr_model="openai/whisper-tiny", embedding_model="bert-base-uncased"):
        self.asr_pipeline = None # Lazy load
        self.asr_model_name = asr_model
        
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embed_model = AutoModel.from_pretrained(embedding_model)
        self.embed_model.eval()

    def transcribe(self, audio_input, sample_rate=16000, chunk_length_s=30):
        """
        Transcribes audio. Accepts a file path or a raw numpy array.
        Enables chunking for long-form audio.
        """
        if self.asr_pipeline is None:
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition", 
                model=self.asr_model_name,
                chunk_length_s=chunk_length_s,
                generate_kwargs={"language": "english", "task": "transcribe"}
            )
        
        # If input is a path, load it with librosa to avoid ffmpeg backend issues in transformers
        if isinstance(audio_input, (str, Path)):
            audio_input, _ = librosa.load(str(audio_input), sr=sample_rate)
            
        result = self.asr_pipeline(audio_input)
        return result["text"]

    def embed(self, text, max_length=200):
        """
        Converts text to BERT embeddings.
        Output dimension is 768 by default for BERT-base.
        (The text branch in models/text_branch.py expects 300 by default, 
        so we may need to adjust or use a different model).
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
        
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
            
        # Use simple tokens embeddings sequence
        # Shape: (1, max_length, hidden_dim)
        embeddings = outputs.last_hidden_state
        return embeddings

if __name__ == "__main__":
    # Test
    # prep = TextPreprocessor()
    # emb = prep.embed("This is a test transcription.")
    # print(f"Text embedding shape: {emb.shape}")
    pass
