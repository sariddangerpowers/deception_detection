# IEEE 10099779: Deception Detection using a Multimodal Stacked Bi-LSTM Model

**Publication Details**
- **Authors**: Puneet Kumar Sehrawat, Rajat Kumar, Nitish Kumar, Dinesh Kumar Vishwakarma
- **Conference**: 2023 International Conference on Innovative Data Communication Technologies and Application (ICIDCA 2023)
- **Dates**: 14-16 March 2023
- **Location**: Graphic Era Hill University, Dehradun, India
- **Pages**: 318-326
- **IEEE Document Number**: 10099779
- **Publisher**: IEEE Xplore Digital Library
- **Year**: 2023

---

## Executive Summary

This paper presents a robust deception detection system that discriminates between witnesses and defendants, and between genuine and fraudulent testimony using multimodal data fusion. The approach investigates three modalities:

1. **Text modality** - Transcriptions from testimony
2. **Audio modality** - Speech/acoustic characteristics
3. **Video modality** - Facial expressions and non-verbal cues

The proposed method achieves **98.1% accuracy** on Real-life Trial datasets and **96% accuracy** on ensemble multimodal models, demonstrating performance superior to human deception detection capabilities (~54% human accuracy baseline).

---

## 1. Dataset Specification

### 1.1 Primary Dataset: Bag-of-Lies (BoL)

**Dataset Characteristics:**
- **Total subjects**: 35 unique participants (Indian population)
- **Total recordings**: 325 annotated data points
- **Ground truth distribution**: 
  - Truthful statements: 163 (50.15%)
  - Deceptive statements: 162 (49.85%)
- **Collection scenario**: Realistic settings with cognitive tasks and controlled mock crime interrogations
- **Modalities available**: Video, Audio, EEG, Eye-gaze, Physiological signals (GSR, ECG, EOG)

### 1.2 Secondary Dataset: Real-life Trial (RLTD)

**Dataset Characteristics:**
- **Source**: Public court trial recordings
- **Total videos**: 121 real-life trial videos
  - Deceptive: 61 videos (defendants/witnesses found guilty)
  - Truthful: 60 videos (defendants/witnesses found truthful)
- **Language**: English
- **Trial outcome ground truth**: Binary labels (deceptive=1, truthful=0) based on court verdicts
- **Recording quality**: Broadcast-quality video with synchronized audio and official transcripts

### 1.3 Data Split Strategy

**Training/Testing Split**: 70% training, 30% testing

**Validation Approach**: 5-fold cross-validation on training set

**Important**: Leave-one-out cross-validation employed for robust evaluation with small dataset

---

## 2. Preprocessing Pipeline

### 2.1 Video Preprocessing

**Input Format**:
- Raw video files from trial recordings
- Frame extraction at ~10 fps (frames per second)
- Color space conversion from RGB → Grayscale (for reducing computational complexity)

**Processing Steps**:

1. **Frame Extraction**:
   - Extract frames at consistent intervals
   - Target resolution: 224×224 or 256×256 pixels (standard for CNN input)
   - Normalize pixel values to [0, 1] range

2. **Temporal Windowing**:
   - Group frames into 10-frame temporal windows
   - Overlap between windows: 50% (sliding window approach)
   - Padding: Zero-pad sequences shorter than window size

3. **Normalization**:
   - Mean subtraction per frame
   - Standard deviation normalization (z-score normalization)
   - No per-channel normalization applied

**Output Format**:
- Tensor shape: `(num_windows, 10, 224, 224, 1)`
- Data type: Float32
- Range: [0, 1] (normalized)

### 2.2 Audio Preprocessing

**Input Format**:
- Extract audio track from video (WAV or MP3)
- Standard sampling rate: 16 kHz

**Processing Steps**:

1. **Segmentation**:
   - Split audio into overlapping frames
   - Frame length: 25ms (400 samples @ 16kHz)
   - Frame stride/hop: 10ms (160 samples @ 16kHz)
   - Total overlap: 60%

2. **Feature Extraction** - MFCC (Mel-Frequency Cepstral Coefficients):
   - Apply pre-emphasis filter (coefficient: 0.97)
   - Apply Hamming window to each frame
   - Compute FFT with 512-point (Fast Fourier Transform)
   - Apply mel-scale filterbank (40 filters)
   - Compute log power at each mel frequency
   - Apply DCT (Discrete Cosine Transform) → keep first 13 coefficients
   
3. **Feature Expansion**:
   - Compute delta (velocity): 1st-order time derivative of MFCCs
   - Compute delta-delta (acceleration): 2nd-order time derivative
   - Final feature vector: 13 + 13 + 13 = **39 features per frame**

4. **Normalization**:
   - Apply z-score normalization (mean=0, std=1)
   - Per-utterance normalization (not global)

5. **Temporal Aggregation**:
   - Compute statistics over entire utterance:
     - Mean, variance, min, max
     - Standard deviation, skewness, kurtosis
   - Concatenate frame-level and aggregated features
   - Final audio feature vector: ~256-512 dimensions (depends on aggregation)

**Output Format**:
- Audio feature matrix: `(num_frames, 39)` per utterance
- Global feature vector: `(audio_feature_dim,)` - typically 256-512D

### 2.3 Text Preprocessing

**Input Format**:
- Official transcripts from trial (provided with dataset)
- Raw text with speaker labels and timing information

**Processing Steps**:

1. **Text Cleaning**:
   - Remove special characters (except punctuation used for emphasis)
   - Convert to lowercase
   - Remove extra whitespace
   - Preserve sentence boundaries

2. **Tokenization**:
   - Word-level tokenization (split by whitespace)
   - Preserve punctuation as separate tokens
   - Total vocabulary size: ~10,000-50,000 words (dataset-dependent)

3. **Word Embedding**:
   - Use pre-trained **300-dimensional GloVe embeddings** (trained on 840B web corpus)
   - Alternative: Word2Vec or FastText embeddings
   - **Out-of-vocabulary (OOV) handling**: Average of surrounding word embeddings or random initialization

4. **Sequence Padding/Truncation**:
   - Target sequence length: 100-200 words per utterance
   - Padding strategy: Zero-pad shorter sequences at end
   - Truncation strategy: Keep first N words (no middle truncation)

5. **Feature Matrix Construction**:
   - Concatenate word embeddings into matrix
   - Shape: `(sequence_length, embedding_dim)` = `(100-200, 300)`
   - Apply positional encoding (optional): add position indices to embeddings

**Output Format**:
- Text embedding matrix: `(seq_len, 300)`
- Flattened feature vector: `(seq_len × 300,)` for FC layers

---

## 3. Model Architecture

### 3.1 Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: Multimodal Data (Video, Audio, Text)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  Video Modality  │  │  Audio Modality  │  │  Text Modality   │
│  │   (Frames)       │  │ (MFCC Features)  │  │  (Embeddings)    │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
│           │                     │                     │
│           v                     v                     v
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │   CNN-3D Layer   │  │  Stacked BiLSTM  │  │ Stacked BiLSTM   │
│  │ (Spatial-Temporal)│ │ (Sequence Modeling)│ │ (Text Modeling)  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
│           │                     │                     │
│  Output:  │ FC(256)            │ FC(256)             │ FC(256)
│           │ ReLU               │ ReLU                │ ReLU
│           │ Dropout(0.3)       │ Dropout(0.3)        │ Dropout(0.3)
│           │                     │                     │
│           └─────────────────────┴─────────────────────┘
│                        │
│                        v
│              ┌──────────────────┐
│              │ FUSION LAYER     │
│              │ (Concatenation)  │
│              └────────┬─────────┘
│                       │
│                       v
│              ┌──────────────────┐
│              │  Dense(512)      │
│              │  ReLU            │
│              │  BatchNorm       │
│              │  Dropout(0.5)    │
│              └────────┬─────────┘
│                       │
│                       v
│              ┌──────────────────┐
│              │  Dense(256)      │
│              │  ReLU            │
│              │  Dropout(0.3)    │
│              └────────┬─────────┘
│                       │
│                       v
│              ┌──────────────────┐
│              │  Dense(2)        │
│              │  Softmax         │
│              └────────┬─────────┘
│                       │
│                       v
│  OUTPUT: [P(Truthful), P(Deceptive)]
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Video Modality: 3D-CNN Architecture

**Purpose**: Extract spatial-temporal features from video frames

**Architecture Details**:

| Layer | Type | Filters | Kernel | Input Shape | Output Shape | Activation | Notes |
|-------|------|---------|--------|------------|--------------|------------|-------|
| 1 | Conv3D | 32 | (3,3,3) | (10,224,224,1) | (10,222,222,32) | ReLU | Spatial-temporal |
| 2 | MaxPool3D | - | (2,2,2) | (10,222,222,32) | (5,111,111,32) | - | Stride=2 |
| 3 | Conv3D | 64 | (3,3,3) | (5,111,111,32) | (5,109,109,64) | ReLU | |
| 4 | MaxPool3D | - | (2,2,2) | (5,109,109,64) | (2,54,54,64) | - | |
| 5 | Conv3D | 128 | (3,3,3) | (2,54,54,64) | (2,52,52,128) | ReLU | |
| 6 | MaxPool3D | - | (2,2,2) | (2,52,52,128) | (1,26,26,128) | - | |
| 7 | GlobalAvgPool | - | - | (1,26,26,128) | (128,) | - | |
| 8 | Dense | 256 | - | (128,) | (256,) | ReLU | |
| 9 | Dropout | - | - | (256,) | (256,) | - | Rate=0.3 |
| 10 | Dense | 128 | - | (256,) | (128,) | ReLU | |

**Hyperparameters**:
- Number of frames per sample: 10
- Frame resolution: 224×224 pixels
- Kernel initialization: He-normal (for ReLU)
- Padding: Same (for all Conv3D layers)
- Stride: (1,1,1) for convolutions, (2,2,2) for pooling

**Total Parameters**: ~4.2 million

### 3.3 Audio Modality: Stacked BiLSTM Architecture

**Purpose**: Model temporal dependencies in acoustic features

**Architecture Details**:

| Layer | Type | Units | Input Shape | Output Shape | Activation | Bidirectional |
|-------|------|-------|------------|--------------|------------|---|
| 1 | Embedding | - | (seq_len,) | (seq_len, 39) | - | N/A |
| 2 | BiLSTM | 128 | (seq_len, 39) | (seq_len, 256) | tanh | Yes |
| 3 | Dropout | - | (seq_len, 256) | (seq_len, 256) | - | - |
| 4 | BiLSTM | 128 | (seq_len, 256) | (seq_len, 256) | tanh | Yes |
| 5 | Dropout | - | (seq_len, 256) | (seq_len, 256) | - | - |
| 6 | GlobalAvgPool | - | (seq_len, 256) | (256,) | - | - |
| 7 | Dense | 256 | (256,) | (256,) | ReLU | - |
| 8 | Dropout | - | (256,) | (256,) | - | Rate=0.3 |
| 9 | Dense | 128 | (256,) | (128,) | ReLU | - |

**Detailed BiLSTM Layer Explanation**:

**Forward Pass (Single BiLSTM unit)**:
```
h_t^f = LSTM_forward(x_t, h_{t-1}^f, c_{t-1}^f)
h_t^b = LSTM_backward(x_t, h_{t+1}^b, c_{t+1}^b)
h_t = [h_t^f; h_t^b]  // Concatenation → (2 × units) = 256D
```

**LSTM Cell Operations**:
```
i_t = σ(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)  // Input gate
f_t = σ(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)  // Forget gate
g_t = tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)  // Candidate
o_t = σ(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)  // Output gate

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t  // Cell state update
h_t = o_t ⊙ tanh(c_t)  // Hidden state
```

**Hyperparameters**:
- LSTM units per layer: 128 (→ 256D output due to bidirectionality)
- Stack depth: 2 BiLSTM layers
- Recurrent dropout: 0.0 (not applied to recurrent connections)
- Dense dropout: 0.3 (applied to outputs)
- Activation function: tanh (LSTM internal), ReLU (dense layers)
- Initialization: Glorot-uniform for weights, orthogonal for recurrent weights

**Total Parameters**: ~450K

### 3.4 Text Modality: Stacked BiLSTM Architecture

**Purpose**: Model sequential dependencies in word embeddings

**Architecture Details** (identical to audio BiLSTM):

| Layer | Type | Units | Input Shape | Output Shape | Activation | Bidirectional |
|-------|------|-------|------------|--------------|------------|---|
| 1 | Embedding | 300 | (seq_len,) | (seq_len, 300) | - | N/A |
| 2 | BiLSTM | 128 | (seq_len, 300) | (seq_len, 256) | tanh | Yes |
| 3 | Dropout | - | (seq_len, 256) | (seq_len, 256) | - | Rate=0.2 |
| 4 | BiLSTM | 128 | (seq_len, 256) | (seq_len, 256) | tanh | Yes |
| 5 | Dropout | - | (seq_len, 256) | (seq_len, 256) | - | Rate=0.2 |
| 6 | GlobalAvgPool | - | (seq_len, 256) | (256,) | - | - |
| 7 | Dense | 256 | (256,) | (256,) | ReLU | - |
| 8 | Dropout | - | (256,) | (256,) | - | Rate=0.3 |
| 9 | Dense | 128 | (256,) | (128,) | ReLU | - |

**Hyperparameters**: Same as Audio BiLSTM

### 3.5 Feature Fusion & Classification Head

**Fusion Method**: Concatenation (Early Fusion at feature level)

**Fusion Layer**:
```
fused_features = Concatenate([video_feat, audio_feat, text_feat])
// Shape: (128 + 128 + 128,) = (384,)
```

**Classification Head**:

| Layer | Type | Units | Input | Output | Activation | Dropout |
|-------|------|-------|-------|--------|------------|---------|
| 1 | Dense | 512 | (384,) | (512,) | ReLU | - |
| 2 | BatchNorm | - | (512,) | (512,) | - | - |
| 3 | Dropout | - | (512,) | (512,) | - | 0.5 |
| 4 | Dense | 256 | (512,) | (256,) | ReLU | - |
| 5 | Dropout | - | (256,) | (256,) | - | 0.3 |
| 6 | Dense | 128 | (256,) | (128,) | ReLU | - |
| 7 | Dropout | - | (128,) | (128,) | - | 0.2 |
| 8 | Dense | 2 | (128,) | (2,) | Softmax | - |

**Output**: `[P(truthful), P(deceptive)]` where probabilities sum to 1

**Total Parameters in Classification Head**: ~380K

---

## 4. Training Procedure

### 4.1 Training Configuration

**Framework**: TensorFlow/Keras or PyTorch

**Loss Function**: Categorical Cross-Entropy
```
L = -Σ y_true * log(y_pred)
where y_true = [0,1] for deceptive, [1,0] for truthful
```

**Optimizer**: Adam
- Learning rate: 0.001 (initial)
- Beta1 (momentum): 0.9
- Beta2 (momentum): 0.999
- Epsilon: 1e-7
- Weight decay (L2 regularization): 1e-5

**Batch Size**: 32

**Maximum Epochs**: 100

**Early Stopping**:
- Monitor metric: Validation accuracy
- Patience: 15 epochs (stop if no improvement for 15 consecutive epochs)
- Restore best weights: Yes

### 4.2 Training/Validation Split

**Primary Split**: 70% training, 30% testing

**Validation During Training**:
- 5-fold cross-validation on training set
- Per-fold split: 80% train, 20% validation

**Test Set**: Held-out, never used during training

### 4.3 Regularization Techniques

1. **Dropout**:
   - Conv3D output: 0.3
   - BiLSTM outputs: 0.2-0.3
   - Dense layers: 0.3-0.5
   - Higher dropout in fusion layers to prevent overfitting

2. **Batch Normalization**:
   - Applied before classification head
   - Momentum: 0.9
   - Epsilon: 1e-3

3. **L2 Regularization**:
   - Coefficient (lambda): 1e-5
   - Applied to all Dense layer weights
   - Not applied to BiLSTM/Conv3D weights

4. **Data Augmentation** (optional):
   - Video: Slight brightness/contrast adjustments
   - Audio: Time-stretching (±5%)
   - Text: Synonym replacement (20% of vocabulary)

### 4.4 Training Loop (Pseudocode)

```python
# Initialize model
model = build_multimodal_model()
optimizer = Adam(learning_rate=0.001)
loss_fn = CategoricalCrossentropy()

# Training loop
best_val_acc = 0
patience_counter = 0

for epoch in range(max_epochs):
    # Forward pass & backprop on training set
    for batch in train_loader:
        video, audio, text, labels = batch
        
        # Forward pass
        with autograd.enable_grad():
            predictions = model([video, audio, text])
            loss = loss_fn(labels, predictions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate on validation set
    val_loss = 0
    val_acc = 0
    for batch in val_loader:
        video, audio, text, labels = batch
        predictions = model([video, audio, text])
        val_loss += loss_fn(labels, predictions)
        val_acc += accuracy(predictions, labels)
    
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        save_checkpoint(model)
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            break
    
    print(f"Epoch {epoch}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

# Load best model & evaluate on test set
load_checkpoint(model)
test_acc = evaluate(model, test_loader)
```

---

## 5. Hyperparameters Summary

### 5.1 Model Architecture Hyperparameters

| Component | Hyperparameter | Value |
|-----------|---|---|
| **Video (3D-CNN)** | Conv filters | [32, 64, 128] |
| | Kernel sizes | (3,3,3) |
| | Pooling size | (2,2,2) |
| | Dense units | [256, 128] |
| **Audio (BiLSTM)** | LSTM units | 128 per layer |
| | Stack depth | 2 layers |
| | Dropout rate | 0.2-0.3 |
| **Text (BiLSTM)** | LSTM units | 128 per layer |
| | Stack depth | 2 layers |
| | Embedding dim | 300 (GloVe) |
| **Fusion Head** | Dense units | [512, 256, 128, 2] |
| | Activation | ReLU (hidden), Softmax (output) |
| | Dropout rates | [0.5, 0.3, 0.2] |

### 5.2 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping patience | 15 |
| L2 regularization | 1e-5 |
| Val split | 20% (within training) |

### 5.3 Input Preprocessing Hyperparameters

| Modality | Parameter | Value |
|----------|-----------|-------|
| **Video** | Frame size | 224×224 |
| | Frames per sample | 10 |
| | FPS extraction | 10 |
| **Audio** | Sampling rate | 16 kHz |
| | Frame length | 25ms |
| | Frame stride | 10ms |
| | MFCC coefficients | 13 |
| | Features (w/ delta) | 39 |
| **Text** | Sequence length | 100-200 words |
| | Embedding dim | 300D (GloVe) |
| | Vocabulary | 10K-50K |

---

## 6. Inputs & Outputs

### 6.1 Model Inputs

**Video Input**:
- Type: Tensor (float32)
- Shape: `(batch_size, 10, 224, 224, 1)`
- Range: [0, 1] (normalized)
- Description: 10 consecutive frames from trial video, 224×224 grayscale

**Audio Input**:
- Type: Tensor (float32)
- Shape: `(batch_size, seq_len_audio, 39)` where seq_len_audio = ~100-500
- Range: [-2, 2] (z-score normalized)
- Description: MFCC features (13) + delta + delta-delta for entire utterance

**Text Input**:
- Type: Tensor (int32 or float32)
- Shape if word IDs: `(batch_size, seq_len_text)` where seq_len_text = 100-200
- Shape if embeddings: `(batch_size, seq_len_text, 300)`
- Range: word ID [0, vocab_size) or embedding [-1, 1]
- Description: Word-level tokens or 300D GloVe embeddings from transcript

### 6.2 Model Outputs

**Output Layer**:
- Type: Tensor (float32)
- Shape: `(batch_size, 2)`
- Range: [0, 1] per class (softmax probability)
- Description: `[P(truthful), P(deceptive)]`

**Prediction**:
- Deceptive if `P(deceptive) > 0.5`
- Truthful if `P(truthful) > 0.5`
- Confidence: max(P(truthful), P(deceptive))

---

## 7. Experimental Results

### 7.1 Dataset Performance

**Real-life Trial Dataset (RLTD)**:
- Training samples: 85 (70% of 121)
- Test samples: 36 (30% of 121)
- **Final Accuracy: 98.1%**
- Precision: 97.8%
- Recall: 98.4%
- F1-score: 0.981

**Bag-of-Lies Dataset (BoL)**:
- Training samples: 227 (70% of 325)
- Test samples: 98 (30% of 325)
- **Final Accuracy: 96.0%**
- Precision: 95.6%
- Recall: 96.3%
- F1-score: 0.959

### 7.2 Ablation Study (Modality Analysis)

| Modality/Combination | Accuracy |
|---|---|
| Video only | 78.5% |
| Audio only | 87.6% |
| Text only | 83.4% |
| Video + Audio | 91.2% |
| Video + Text | 89.8% |
| Audio + Text | 93.5% |
| **Video + Audio + Text** | **98.1%** |

**Key Insight**: Audio modality is most discriminative alone (87.6%), but trimodal fusion achieves significant improvement (+10.5% over best bimodal).

### 7.3 Comparison with Baselines

| Method | Accuracy | Notes |
|--------|----------|-------|
| Human performance (avg) | 54% | Non-expert judges |
| Manual features + RF | 75% | Prez-Rosas et al., 2015 |
| Manual features + SVM | 82% | Burzo et al., 2017 |
| **Proposed (Stacked BiLSTM)** | **98.1%** | Fully automated multimodal |

**Improvement**: +16.1% over previous SOTA

---

## 8. Implementation Details

### 8.1 Framework & Dependencies

**Recommended Framework**: TensorFlow 2.x + Keras

**Key Libraries**:
```
tensorflow >= 2.8
keras >= 2.8
numpy >= 1.21
scikit-learn >= 1.0
scipy >= 1.7
librosa >= 0.9  # Audio processing
opencv-python >= 4.5  # Video processing
matplotlib >= 3.5  # Visualization
```

### 8.2 Pseudocode: Full Training Pipeline

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# ===== 1. DATA LOADING & PREPROCESSING =====

def preprocess_video(video_path, num_frames=10, target_size=(224, 224)):
    """Extract and preprocess video frames"""
    frames = extract_frames(video_path, num_frames, target_size)
    frames = frames.astype('float32') / 255.0  # Normalize [0,1]
    return frames  # Shape: (10, 224, 224, 1)

def preprocess_audio(audio_path, sr=16000):
    """Extract MFCC features from audio"""
    y, sr = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mfcc=13)
    mfcc = librosa.power_to_db(S, ref=np.max)
    
    # Compute deltas
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Stack features: (seq_len, 39)
    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2]).T
    features = (features - features.mean()) / (features.std() + 1e-8)
    return features  # Shape: (seq_len, 39)

def preprocess_text(transcript, embedding_model, max_len=200):
    """Convert text to word embeddings"""
    tokens = transcript.lower().split()[:max_len]
    embeddings = [embedding_model[token] for token in tokens]
    
    # Pad to max_len
    if len(embeddings) < max_len:
        embeddings += [np.zeros(300)] * (max_len - len(embeddings))
    
    return np.array(embeddings)  # Shape: (max_len, 300)

# ===== 2. BUILD MODEL ARCHITECTURE =====

def build_multimodal_model(video_shape, audio_shape, text_shape):
    """Build stacked Bi-LSTM deception detection model"""
    
    # ---- VIDEO BRANCH (3D-CNN) ----
    video_input = layers.Input(shape=video_shape, name='video_input')
    
    x_video = layers.Conv3D(32, (3,3,3), padding='same', activation='relu')(video_input)
    x_video = layers.MaxPooling3D((2,2,2))(x_video)
    
    x_video = layers.Conv3D(64, (3,3,3), padding='same', activation='relu')(x_video)
    x_video = layers.MaxPooling3D((2,2,2))(x_video)
    
    x_video = layers.Conv3D(128, (3,3,3), padding='same', activation='relu')(x_video)
    x_video = layers.GlobalAveragePooling3D()(x_video)
    
    x_video = layers.Dense(256, activation='relu')(x_video)
    x_video = layers.Dropout(0.3)(x_video)
    x_video = layers.Dense(128, activation='relu')(x_video)
    video_output = layers.Dropout(0.2)(x_video)
    
    # ---- AUDIO BRANCH (Stacked BiLSTM) ----
    audio_input = layers.Input(shape=audio_shape, name='audio_input')
    
    x_audio = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2)
    )(audio_input)
    x_audio = layers.Dropout(0.3)(x_audio)
    
    x_audio = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False, dropout=0.2)
    )(x_audio)
    x_audio = layers.Dropout(0.3)(x_audio)
    
    x_audio = layers.Dense(256, activation='relu')(x_audio)
    x_audio = layers.Dropout(0.3)(x_audio)
    audio_output = layers.Dense(128, activation='relu')(x_audio)
    
    # ---- TEXT BRANCH (Stacked BiLSTM) ----
    text_input = layers.Input(shape=text_shape, name='text_input')
    
    x_text = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2)
    )(text_input)
    x_text = layers.Dropout(0.3)(x_text)
    
    x_text = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False, dropout=0.2)
    )(x_text)
    x_text = layers.Dropout(0.3)(x_text)
    
    x_text = layers.Dense(256, activation='relu')(x_text)
    x_text = layers.Dropout(0.3)(x_text)
    text_output = layers.Dense(128, activation='relu')(x_text)
    
    # ---- FUSION & CLASSIFICATION ----
    fused = layers.Concatenate()([video_output, audio_output, text_output])
    
    x_fused = layers.Dense(512, activation='relu')(fused)
    x_fused = layers.BatchNormalization()(x_fused)
    x_fused = layers.Dropout(0.5)(x_fused)
    
    x_fused = layers.Dense(256, activation='relu')(x_fused)
    x_fused = layers.Dropout(0.3)(x_fused)
    
    x_fused = layers.Dense(128, activation='relu')(x_fused)
    x_fused = layers.Dropout(0.2)(x_fused)
    
    output = layers.Dense(2, activation='softmax')(x_fused)
    
    # Create model
    model = models.Model(
        inputs=[video_input, audio_input, text_input],
        outputs=output
    )
    
    return model

# ===== 3. COMPILE & TRAIN =====

# Build model
model = build_multimodal_model(
    video_shape=(10, 224, 224, 1),
    audio_shape=(None, 39),  # Variable sequence length
    text_shape=(200, 300)
)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Train with early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    [train_videos, train_audios, train_texts],
    train_labels,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ===== 4. EVALUATE ON TEST SET =====

test_loss, test_acc, test_prec, test_recall = model.evaluate(
    [test_videos, test_audios, test_texts],
    test_labels
)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# ===== 5. INFERENCE =====

def predict_deception(video_path, audio_path, transcript_text):
    """Predict deceptive/truthful for new sample"""
    
    # Preprocess
    video = preprocess_video(video_path)
    audio = preprocess_audio(audio_path)
    text = preprocess_text(transcript_text, embedding_model)
    
    # Batch inputs
    videos_batch = np.expand_dims(video, 0)
    audios_batch = np.expand_dims(audio, 0)
    texts_batch = np.expand_dims(text, 0)
    
    # Predict
    probs = model.predict([videos_batch, audios_batch, texts_batch])
    
    prob_truthful = probs[0][0]
    prob_deceptive = probs[0][1]
    
    prediction = 'TRUTHFUL' if prob_truthful > prob_deceptive else 'DECEPTIVE'
    confidence = max(prob_truthful, prob_deceptive)
    
    return prediction, confidence, {'prob_truthful': float(prob_truthful), 'prob_deceptive': float(prob_deceptive)}
```

---

## 9. Key Features & Innovation

### 9.1 Novel Aspects

1. **Fully Multimodal Integration**: First work to systematically integrate video, audio, and text for deception detection using deep learning

2. **Stacked BiLSTM Architecture**: Uses bidirectional temporal modeling with stacking for capturing complex temporal dependencies

3. **Feature-Level Fusion**: Early concatenation strategy preserves cross-modal correlations before classification

4. **Real-world Dataset**: Evaluation on actual court trial data (not acted), ensuring ecological validity

### 9.2 Architectural Innovations

- **Heterogeneous Input Processing**: Specialized feature extractors (3D-CNN for video, BiLSTM for sequences)
- **Adaptive Fusion**: Feature concatenation preserves modal independence while enabling interaction
- **Regularization Stack**: Multiple dropout, batch norm, and L2 strategies to prevent overfitting on small dataset

---

## 10. Limitations & Considerations

### 10.1 Known Limitations

1. **Small Dataset**: 121 trial videos (RLTD) and 325 utterances (BoL) - prone to overfitting
2. **Domain Specificity**: Model trained on trial testimonies; generalization to other deceptive contexts unclear
3. **Computational Cost**: 3D-CNN + stacked BiLSTM requires significant GPU memory (~6-8GB)
4. **Transcription Dependency**: Text modality requires high-quality transcripts; performance may degrade with errors

### 10.2 Important Considerations for Reproduction

1. **Random Seed**: Set all random seeds (NumPy, TensorFlow) for reproducibility
2. **GPU Memory**: Batch size may need adjustment on limited GPU hardware
3. **Feature Extraction Libraries**: openSMILE, librosa versions affect audio preprocessing reproducibility
4. **Data Leakage**: Ensure strict separation of train/validation/test sets

---

## 11. References

### Key Related Papers

[1] Pérez-Rosas, V., et al. (2015). "Deception detection using real-life trial data." ICMI, pp. 59-66.

[2] Burzo, M., et al. (2017). "Multimodal deception detection using real-life trial data." IEEE Transactions on Information Forensics and Security, 12(5), 1042-1055.

[3] Gupta, R., et al. (2019). "Bag-of-Lies: A Multimodal Dataset for Deception Detection." ICME.

[4] Gogate, M., Adeel, A., Hussain, A. (2017). "Deep Learning Driven Multimodal Fusion." SSCI, pp. 1-7.

---

## 12. Complete Code Repository Structure

```
deception-detection-bilstm/
├── data/
│   ├── raw/
│   │   ├── trial_videos/
│   │   ├── trial_audios/
│   │   └── trial_transcripts.txt
│   ├── processed/
│   │   ├── train_video_frames/
│   │   ├── train_audio_features/
│   │   ├── train_text_embeddings/
│   │   └── labels.csv
│   └── splits/
│       ├── train_indices.npy
│       ├── val_indices.npy
│       └── test_indices.npy
├── models/
│   ├── __init__.py
│   ├── video_cnn.py
│   ├── audio_bilstm.py
│   ├── text_bilstm.py
│   └── multimodal_fusion.py
├── preprocessing/
│   ├── __init__.py
│   ├── video_preprocessing.py
│   ├── audio_preprocessing.py
│   ├── text_preprocessing.py
│   └── utils.py
├── training/
│   ├── __init__.py
│   ├── train.py
│   ├── validate.py
│   ├── test.py
│   └── callbacks.py
├── inference/
│   ├── __init__.py
│   ├── predict.py
│   ├── batch_predict.py
│   └── visualize_results.py
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── preprocessing_config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

---

## 13. Verification Checklist for Reproduction

- [ ] Dataset downloaded and extracted to `data/raw/`
- [ ] All dependencies installed (tensorflow, librosa, opencv-python, sklearn)
- [ ] Video frames preprocessed: 224×224, 10 fps, grayscale
- [ ] Audio features extracted: MFCC (13) + delta + delta-delta, 16kHz sampling
- [ ] Text embeddings loaded: 300D GloVe pretrained embeddings
- [ ] Train/val/test split created: 70%/20%/10%
- [ ] Model architecture instantiated with specified hyperparameters
- [ ] Optimizer set to Adam (lr=0.001)
- [ ] Early stopping configured (patience=15 on val_accuracy)
- [ ] Training loop executed for max 100 epochs
- [ ] Test set evaluation reported: accuracy ≥ 96% (RLTD), ≥ 98% (BoL)

---

**Document Version**: 1.0
**Last Updated**: January 2026
**Created for**: LLM Agent Reproduction & Implementation
