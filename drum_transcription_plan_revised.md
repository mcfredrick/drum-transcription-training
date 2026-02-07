# Drum Transcription Model Implementation Plan (REVISED)
## For Electronic Drum Rhythm Game - 8-Lane System

---

## Executive Summary - Key Changes

Based on your feedback, this revised plan makes the following improvements:

1. **Primary Dataset: E-GMD** - Perfect class alignment with your 8-lane system (kick, snare, hi-hat, hi-tom, mid-tom, low-tom, crash, ride)
2. **No Stem Separation for Training** - Train directly on isolated drums from E-GMD; only use Demucs for deployment if needed
3. **PyTorch Lightning** - Most ergonomic framework for occasional ML work, eliminates boilerplate
4. **UV for Dependency Management** - Modern, fast Python package manager with great PyTorch support
5. **Feature Extraction IS Necessary** - Raw audio → spectrograms (this is standard preprocessing)

**Timeline**: 4-6 weeks for PoC with cleaner path forward

---

## 1. Why E-GMD is Perfect for Your Use Case

### E-GMD Dataset Advantages

**Perfect Class Matching:**
```
E-GMD Classes          →  Your 8-Lane System
─────────────────────     ──────────────────
Kick (Bass Drum)      →   Kick ✓
Snare                 →   Snare ✓
Hi-hat (Closed)       →   Hi-hat ✓
High Tom              →   Hi-Tom ✓
Mid Tom               →   Mid-Tom ✓
Low Tom               →   Low-Tom ✓
Crash Cymbal          →   Crash ✓
Ride Cymbal           →   Ride ✓
Floor Tom             →   Low-Tom (merge) ✓
```

**Dataset Characteristics:**
- **444 hours** of drum performances
- **43 different drum kits** (diverse timbres)
- **Isolated drum recordings** (no melodic instruments interfering)
- **High-quality MIDI alignment** (<2ms precision)
- **Roland TD-17** electronic kit recordings
- **Human performances** (natural dynamics and timing)
- **Genre-diverse** (rock, jazz, funk, latin, etc.)
- **Freely available** (CC BY 4.0 license)

### Why This Simplifies Your Workflow

1. **No post-processing needed** - Direct 8-class output
2. **No stem separation for training** - Already isolated drums
3. **Clean ground truth** - Professional MIDI annotations
4. **Better generalization** - 43 kits means diverse timbres
5. **Closer to your use case** - Electronic drums similar to user uploads

---

## 2. Understanding Feature Extraction (Audio → Spectrograms)

### Why Feature Extraction is Necessary

**The Short Answer:** Neural networks need structured numerical input. Raw audio waveforms are too high-dimensional and variable for efficient training.

**The Process:**
```
Raw Audio (WAV/MP3)
    ↓
[Feature Extraction]
    ↓
Log-Mel Spectrogram (image-like representation)
    ↓
[CRNN Model]
    ↓
Frame-level predictions (8 classes)
    ↓
[Post-processing]
    ↓
MIDI output
```

### What is a Log-Mel Spectrogram?

Think of it as an "image" of sound:
- **X-axis**: Time (seconds)
- **Y-axis**: Frequency (Hz), weighted toward human hearing (mel scale)
- **Color**: Loudness (decibels)

**Example:**
- Kick drum: Low frequency, short duration → bright spot at bottom
- Hi-hat: High frequency, short duration → bright spot at top
- Snare: Broad frequency, short duration → bright spot across middle

### Feature Extraction Code (What You'll Actually Run)

```python
import librosa
import numpy as np

def extract_features(audio_path):
    """
    Convert audio file to log-mel spectrogram
    This is what the model will "see"
    """
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=22050)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=2048,      # Window size
        hop_length=512,  # Stride (controls time resolution)
        n_mels=128,      # Number of frequency bands
        fmin=30,         # Minimum frequency (captures kick)
        fmax=11025       # Maximum frequency (Nyquist)
    )
    
    # Convert to log scale (decibels)
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    return log_mel_spec  # Shape: (128, time_frames)
```

**This preprocessing happens once per audio file and is cached.**

### E-GMD Annotations

E-GMD comes with MIDI files that are already aligned to the audio. You'll convert these to frame-level labels:

```python
import pretty_midi

def midi_to_labels(midi_path, num_frames, hop_length=512, sr=22050):
    """
    Convert MIDI annotations to frame-level labels
    Each frame gets 8 binary values (one per drum)
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    
    # Initialize empty labels
    labels = np.zeros((num_frames, 8))
    
    # E-GMD drum mapping (General MIDI)
    drum_map = {
        36: 0,  # Kick
        38: 1,  # Snare
        42: 2,  # Hi-hat (closed)
        50: 3,  # High Tom
        47: 4,  # Mid Tom
        45: 5,  # Low Tom (or 41: Floor Tom)
        49: 6,  # Crash
        51: 7,  # Ride
    }
    
    for instrument in midi.instruments:
        if instrument.is_drum:
            for note in instrument.notes:
                if note.pitch in drum_map:
                    # Convert time to frame index
                    frame = int(note.start * sr / hop_length)
                    
                    # Mark onset (with ±2 frame tolerance)
                    drum_idx = drum_map[note.pitch]
                    labels[max(0, frame-2):min(num_frames, frame+3), drum_idx] = 1
    
    return labels  # Shape: (num_frames, 8)
```

**Summary**: Feature extraction converts audio to spectrograms (what the model sees), and MIDI labels are converted to frame-level targets (what the model learns to predict).

---

## 3. Framework Choice: PyTorch Lightning

### Why PyTorch Lightning?

Based on research, **PyTorch Lightning is the best choice for occasional ML work**:

**Advantages:**
1. **Eliminates boilerplate** - No manual training loops
2. **Cleaner code** - Structured, modular approach
3. **Built-in features** - Checkpointing, logging, early stopping
4. **Easy multi-GPU** - Just change a flag
5. **Same performance** - Within 0.06s/epoch of raw PyTorch
6. **Industry standard** - 28k GitHub stars, used by Stable Diffusion
7. **Better debugging** - Still uses standard Python debuggers

**Perfect for your case:**
- You work with ML occasionally (not full-time)
- You want clean, maintainable code
- You have 2 GPUs (Lightning makes multi-GPU trivial)
- You want to focus on the model, not infrastructure

### PyTorch Lightning vs Raw PyTorch

**Raw PyTorch (Lots of boilerplate):**
```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    # Validation loop (20+ more lines)
    # Checkpointing (10+ more lines)
    # Logging (10+ more lines)
```

**PyTorch Lightning (Clean and simple):**
```python
# Define model
class DrumTranscriber(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Train
trainer = pl.Trainer(max_epochs=100, devices=2)
trainer.fit(model, train_loader, val_loader)
```

**That's it. Lightning handles everything else automatically.**

---

## 4. Dependency Management with UV

### Why UV?

UV is the modern, fast Python package manager built in Rust:
- **10-100x faster** than pip
- **Better dependency resolution**
- **Built-in virtual environment management**
- **Great PyTorch CUDA support**

### Setting Up UV on macOS + Linux (Your Training Machines)

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart terminal, then verify
uv --version
```

### Project Setup with UV

```bash
# Create new project
mkdir drum_transcription
cd drum_transcription
uv init --python 3.12  # PyTorch works best with 3.12

# This creates:
# - pyproject.toml (dependencies)
# - .python-version (Python version)
# - .venv (virtual environment, auto-created)
```

### Installing PyTorch with CUDA using UV

**For your training machines (3070 + 3090 with CUDA):**

Edit `pyproject.toml`:

```toml
[project]
name = "drum-transcription"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.4.0",
    "torchvision>=0.19.0",
    "lightning>=2.4.0",  # PyTorch Lightning
    "librosa>=0.10.0",
    "soundfile>=0.12.1",
    "mido>=1.3.0",
    "pretty-midi>=0.2.10",
    "wandb>=0.15.0",  # Experiment tracking
    "matplotlib>=3.8.0",
    "numpy>=1.26.0",
    "tqdm>=4.65.0",
]

# PyTorch CUDA configuration
[tool.uv.sources]
torch = [
    { index = "pytorch-cu121", marker = "sys_platform == 'linux'" }
]
torchvision = [
    { index = "pytorch-cu121", marker = "sys_platform == 'linux'" }
]

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
```

**For macOS (development/preprocessing):**

macOS doesn't support CUDA, so it will use CPU version automatically:

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cu121", marker = "sys_platform == 'linux'" },
    # macOS will use PyPI (CPU version) automatically
]
torchvision = [
    { index = "pytorch-cu121", marker = "sys_platform == 'linux'" },
]
```

### Installing Dependencies

```bash
# Install all dependencies (UV handles everything)
uv sync

# Verify PyTorch CUDA installation
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should output: CUDA available: True (on Linux with GPU)
```

### Adding New Dependencies

```bash
# Add packages easily
uv add scikit-learn
uv add tensorboard

# Remove packages
uv remove tensorboard
```

### Common UV Tricks for macOS Installation Issues

**If you encounter build issues on macOS:**

1. **Install Xcode Command Line Tools:**
```bash
xcode-select --install
```

2. **Install Homebrew dependencies:**
```bash
brew install libsndfile  # For soundfile
brew install portaudio   # For audio libraries
```

3. **Use UV's compiled wheels when possible:**
```bash
# UV automatically prefers wheels over source builds
# This avoids compilation issues on macOS
```

4. **If a package fails, try installing without extras:**
```bash
uv add librosa --no-build-isolation
```

---

## 5. Model Architecture: CRNN with PyTorch Lightning

### Complete Model Implementation

```python
import torch
import torch.nn as nn
import lightning as L
from torch.nn import functional as F

class DrumTranscriptionCRNN(L.LightningModule):
    """
    CRNN model for drum transcription
    Implements PyTorch Lightning interface for clean training
    """
    
    def __init__(
        self,
        n_mels=128,
        n_classes=8,
        learning_rate=1e-3,
        hidden_size=128,
        num_gru_layers=3
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        
        # CNN Encoder
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),
        )
        
        # Calculate CNN output size
        self.cnn_output_size = 128 * (n_mels // 8)
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_gru_layers > 1 else 0
        )
        
        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
            nn.Sigmoid()
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
    
    def forward(self, x):
        """Forward pass through the model"""
        batch_size = x.size(0)
        
        # CNN encoding: (batch, 1, n_mels, time)
        x = self.conv_blocks(x)
        
        # Reshape for RNN: (batch, time, features)
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, x.size(1), -1)
        
        # RNN processing
        x, _ = self.gru(x)  # (batch, time, hidden*2)
        
        # Frame-level predictions
        x = self.fc(x)  # (batch, time, n_classes)
        
        return x
    
    def training_step(self, batch, batch_idx):
        """Training step (called automatically by Lightning)"""
        x, y = batch
        y_hat = self(x)
        
        # Compute loss
        loss = self.criterion(y_hat, y)
        
        # Log metrics (Lightning handles this)
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step (called automatically by Lightning)"""
        x, y = batch
        y_hat = self(x)
        
        # Compute loss
        loss = self.criterion(y_hat, y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
```

### Dataset Implementation with PyTorch Lightning

```python
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

class E_GMD_Dataset(Dataset):
    """
    E-GMD dataset for drum transcription
    Loads preprocessed spectrograms and MIDI labels
    """
    
    def __init__(self, audio_files, midi_files, n_mels=128):
        self.audio_files = audio_files
        self.midi_files = midi_files
        self.n_mels = n_mels
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            x: Log-mel spectrogram (1, n_mels, time)
            y: Frame-level labels (time, 8)
        """
        # Load preprocessed features
        spec = self.load_spectrogram(self.audio_files[idx])
        labels = self.load_labels(self.midi_files[idx], spec.shape[-1])
        
        # Convert to tensors
        spec = torch.FloatTensor(spec).unsqueeze(0)  # Add channel dim
        labels = torch.FloatTensor(labels)
        
        return spec, labels
    
    def load_spectrogram(self, audio_path):
        """Load pre-computed spectrogram (implement based on storage)"""
        # This would load from your preprocessed .npy or .h5 files
        import numpy as np
        return np.load(audio_path.replace('.wav', '_spec.npy'))
    
    def load_labels(self, midi_path, num_frames):
        """Load MIDI labels converted to frame-level"""
        import numpy as np
        return np.load(midi_path.replace('.mid', '_labels.npy'))


class DrumDataModule(L.LightningDataModule):
    """
    Lightning DataModule for E-GMD
    Handles all data loading logic
    """
    
    def __init__(
        self,
        train_files,
        val_files,
        test_files,
        batch_size=16,
        num_workers=4
    ):
        super().__init__()
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        """Create datasets"""
        if stage == 'fit' or stage is None:
            self.train_dataset = E_GMD_Dataset(
                self.train_files['audio'],
                self.train_files['midi']
            )
            self.val_dataset = E_GMD_Dataset(
                self.val_files['audio'],
                self.val_files['midi']
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = E_GMD_Dataset(
                self.test_files['audio'],
                self.test_files['midi']
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
```

### Training Script (Super Simple with Lightning)

```python
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

# Initialize model
model = DrumTranscriptionCRNN(
    n_mels=128,
    n_classes=8,
    learning_rate=1e-3,
    hidden_size=128,
    num_gru_layers=3
)

# Initialize data module
data_module = DrumDataModule(
    train_files=train_files,
    val_files=val_files,
    test_files=test_files,
    batch_size=16,
    num_workers=4
)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='drum-transcription-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min'
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=20,
    mode='min',
    verbose=True
)

# Logger (Weights & Biases)
wandb_logger = WandbLogger(project='drum-transcription', name='crnn-egmd')

# Trainer (this is where Lightning shines!)
trainer = L.Trainer(
    max_epochs=100,
    devices=[0, 1],  # Use both GPUs!
    accelerator='gpu',
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=wandb_logger,
    precision='16-mixed',  # Automatic mixed precision
    gradient_clip_val=1.0,
    log_every_n_steps=10
)

# Train (one line!)
trainer.fit(model, data_module)

# Test
trainer.test(model, data_module)
```

**That's it! No manual training loops, no device management, no boilerplate.**

---

## 6. Data Augmentation Strategy

### Augmentations for Drum Transcription

```python
import torch
import torchaudio.transforms as T

class DrumAugmentation:
    """Data augmentation for drum spectrograms"""
    
    def __init__(self, sr=22050):
        self.sr = sr
        
        # Time stretch
        self.time_stretch = T.TimeStretch(
            hop_length=512,
            n_freq=128
        )
        
        # Pitch shift
        self.pitch_shift = T.PitchShift(
            sample_rate=sr,
            n_steps=[-2, -1, 0, 1, 2]  # ±2 semitones
        )
    
    def __call__(self, spec, labels):
        """Apply random augmentation"""
        import random
        
        # 1. Time stretching (0.9x - 1.1x)
        if random.random() < 0.5:
            rate = random.uniform(0.9, 1.1)
            spec = self.time_stretch(spec, rate)
            # Adjust labels accordingly
            labels = self.adjust_labels_for_time_stretch(labels, rate)
        
        # 2. Pitch shifting (±2 semitones)
        if random.random() < 0.5:
            n_steps = random.choice([-2, -1, 0, 1, 2])
            spec = self.pitch_shift(spec, n_steps)
        
        # 3. Volume scaling (±6dB)
        if random.random() < 0.5:
            gain = random.uniform(-6, 6)
            spec = spec + gain
        
        # 4. Reverb (your addition!)
        if random.random() < 0.3:
            spec = self.apply_reverb(spec)
        
        # 5. Background noise
        if random.random() < 0.3:
            noise_level = random.uniform(0.01, 0.05)
            noise = torch.randn_like(spec) * noise_level
            spec = spec + noise
        
        return spec, labels
    
    def apply_reverb(self, spec):
        """
        Simple reverb simulation by adding delayed, attenuated copies
        This simulates room acoustics
        """
        import torch.nn.functional as F
        
        # Small room (10-30ms delay)
        delay = random.randint(1, 3)  # frames
        attenuation = random.uniform(0.2, 0.4)
        
        # Pad and shift
        reverb = F.pad(spec, (delay, 0), mode='constant', value=0)[:, :, :-delay]
        
        return spec + (reverb * attenuation)
    
    def adjust_labels_for_time_stretch(self, labels, rate):
        """Adjust label timing after time stretch"""
        import torch.nn.functional as F
        
        new_length = int(labels.shape[0] / rate)
        labels = F.interpolate(
            labels.unsqueeze(0).unsqueeze(0),
            size=new_length,
            mode='nearest'
        ).squeeze()
        
        return labels
```

---

## 7. Stem Separation: When and How

### Decision: When to Use Stem Separation

**Training (E-GMD):** ❌ **NO stem separation needed**
- E-GMD is already isolated drums
- Train directly on clean audio

**Deployment (User uploads):** ⚠️ **OPTIONAL, test first**

**Recommendation:**
1. **Phase 1**: Train model on E-GMD without stem separation
2. **Phase 2**: Test trained model on full mixes (user uploads)
3. **Phase 3**: Only add Demucs if accuracy is insufficient on full mixes

### If You Need Stem Separation (Demucs v4)

```bash
# Install Demucs
uv add demucs

# Separate drums from full mix
demucs --two-stems=drums user_upload.mp3

# Output: user_upload/drums.wav
```

### Integration Code (If Needed)

```python
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio

def separate_drums(audio_path):
    """
    Extract drum stem from full mix using Demucs
    Only use if model performs poorly on full mixes
    """
    # Load Demucs model
    model = get_model('htdemucs_ft')
    model.eval()
    
    # Load audio
    wav, sr = torchaudio.load(audio_path)
    
    # Separate
    sources = apply_model(model, wav.unsqueeze(0))
    
    # Extract drums (index 2 for htdemucs_ft)
    drums = sources[0, 2]  # (channels, samples)
    
    return drums, sr
```

**Performance note:** Demucs takes ~30-60 seconds for a 3-minute song on GPU.

---

## 8. Complete Workflow Pipeline

### End-to-End System

```
User Upload (MP3/WAV)
    ↓
[Optional: Demucs Stem Separation if needed]
    ↓
[Feature Extraction: Audio → Log-Mel Spectrogram]
    ↓
[CRNN Model: Spectrogram → Frame Predictions]
    ↓
[Post-Processing: Peaks → Onsets]
    ↓
[MIDI Export: General MIDI Drum Mapping]
    ↓
Game-Ready MIDI File (Cached)
```

### Inference Code

```python
import torch
import librosa
import numpy as np
from pathlib import Path

class DrumTranscriber:
    """
    End-to-end drum transcription pipeline
    Handles: loading audio → transcription → MIDI export
    """
    
    def __init__(self, model_path, use_stem_separation=False):
        self.model = DrumTranscriptionCRNN.load_from_checkpoint(model_path)
        self.model.eval()
        self.use_stem_separation = use_stem_separation
        
        if use_stem_separation:
            from demucs.pretrained import get_model
            self.demucs = get_model('htdemucs_ft')
            self.demucs.eval()
    
    @torch.no_grad()
    def transcribe(self, audio_path, output_midi_path):
        """
        Transcribe audio file to MIDI
        
        Args:
            audio_path: Path to audio file
            output_midi_path: Where to save MIDI output
        """
        # 1. Load audio (with optional stem separation)
        if self.use_stem_separation:
            audio = self.separate_drums(audio_path)
        else:
            audio, sr = librosa.load(audio_path, sr=22050)
        
        # 2. Extract features
        spec = self.extract_features(audio)
        
        # 3. Run model
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)  # (1, 1, freq, time)
        predictions = self.model(spec_tensor)  # (1, time, 8)
        
        # 4. Post-process predictions → onsets
        onsets = self.postprocess_predictions(predictions[0].cpu().numpy())
        
        # 5. Export to MIDI
        self.export_midi(onsets, output_midi_path)
        
        return onsets
    
    def extract_features(self, audio, sr=22050):
        """Extract log-mel spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            fmin=30,
            fmax=11025
        )
        log_mel_spec = librosa.power_to_db(mel_spec)
        return log_mel_spec
    
    def postprocess_predictions(self, predictions, threshold=0.5, min_interval=0.05):
        """
        Convert frame-level probabilities to discrete onsets
        
        Args:
            predictions: (time, 8) array of probabilities
            threshold: Minimum probability for detection
            min_interval: Minimum time between onsets (seconds)
        """
        from scipy.signal import find_peaks
        
        onsets = []
        frame_rate = 22050 / 512  # ~43 FPS
        
        drum_names = ['kick', 'snare', 'hihat', 'hi_tom', 'mid_tom', 'low_tom', 'crash', 'ride']
        
        for class_idx, drum_name in enumerate(drum_names):
            # Threshold
            class_preds = predictions[:, class_idx] > threshold
            
            # Find peaks
            peaks, _ = find_peaks(class_preds.astype(float), distance=int(min_interval * frame_rate))
            
            for peak in peaks:
                onset_time = peak / frame_rate
                onsets.append({
                    'time': onset_time,
                    'drum': drum_name,
                    'drum_idx': class_idx,
                    'velocity': 80  # Fixed velocity for now
                })
        
        return sorted(onsets, key=lambda x: x['time'])
    
    def export_midi(self, onsets, output_path, tempo=120):
        """Export onsets to General MIDI drum file"""
        import mido
        
        # General MIDI drum mapping
        GM_DRUM_MAP = {
            'kick': 36,
            'snare': 38,
            'hihat': 42,
            'hi_tom': 50,
            'mid_tom': 47,
            'low_tom': 45,
            'crash': 49,
            'ride': 51,
        }
        
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Set tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
        
        # Convert onsets to MIDI notes
        current_time = 0
        for onset in onsets:
            # Calculate time delta in ticks
            onset_ticks = int(onset['time'] * mid.ticks_per_beat * (tempo / 60))
            delta_ticks = onset_ticks - current_time
            
            note = GM_DRUM_MAP[onset['drum']]
            velocity = onset['velocity']
            
            # Note on
            track.append(mido.Message(
                'note_on',
                note=note,
                velocity=velocity,
                time=delta_ticks,
                channel=9  # Channel 10 (0-indexed as 9)
            ))
            
            # Note off (short duration)
            track.append(mido.Message(
                'note_off',
                note=note,
                velocity=0,
                time=50,  # 50 ticks (~100ms)
                channel=9
            ))
            
            current_time = onset_ticks + 50
        
        mid.save(output_path)

# Usage
transcriber = DrumTranscriber(
    model_path='checkpoints/best_model.ckpt',
    use_stem_separation=False  # Start without, add if needed
)

transcriber.transcribe('user_song.mp3', 'output.mid')
```

---

## 9. Implementation Timeline (Revised)

### Week 1: Setup & Data Preparation

**Days 1-2: Environment Setup**
- Install UV on all machines (macOS + Linux)
- Create project with `uv init`
- Configure `pyproject.toml` for PyTorch CUDA
- Install dependencies with `uv sync`
- Test PyTorch + CUDA installation
- Download E-GMD dataset (90GB) - run overnight

**Days 3-4: Data Exploration**
- Explore E-GMD structure
- Verify MIDI annotations
- Test audio loading with librosa
- Test MIDI parsing with pretty_midi
- Visualize spectrograms
- Check class distribution

**Days 5-7: Preprocessing Pipeline**
- Implement feature extraction (audio → spectrogram)
- Implement MIDI → frame labels conversion
- Batch process E-GMD dataset
- Save preprocessed data (spectrograms + labels as .npy or .h5)
- Create train/val/test splits (70/15/15)
- Verify data quality

**Deliverable:** Preprocessed E-GMD dataset ready for training

### Week 2: Baseline Model

**Days 8-9: Model Implementation**
- Implement CRNN model in PyTorch Lightning
- Implement E-GMD Dataset class
- Implement DataModule
- Test on small subset (overfitting test)

**Days 10-12: Training**
- Train baseline model (no augmentation)
- Monitor with Weights & Biases
- Evaluate on validation set
- Debug any issues

**Days 13-14: Initial Evaluation**
- Implement post-processing (peaks → onsets)
- Implement MIDI export
- Compute F-measure on test set
- Listen to results (qualitative check)

**Expected Results:**
- F-measure: 60-70% (baseline, no augmentation)
- Clear detection of kick, snare, hi-hat
- Some confusion on toms and cymbals

**Deliverable:** Working baseline model

### Week 3: Optimization

**Days 15-17: Data Augmentation**
- Implement augmentation pipeline
- Add: time stretch, pitch shift, volume, reverb, noise
- Retrain with augmentation
- Compare results

**Days 18-19: Hyperparameter Tuning**
- Learning rate experiments
- GRU layer count (2, 3, 4)
- Hidden dimension size (64, 128, 256)
- Dropout rates
- Use Lightning's built-in hyperparameter search if time permits

**Days 20-21: Model Selection**
- Train best configuration
- Extended training (100+ epochs with early stopping)
- Final evaluation on test set

**Expected Results:**
- F-measure: 70-80%
- Better generalization
- Fewer false positives

**Deliverable:** Optimized model with 70-80% F-measure

### Week 4: Real-World Testing

**Days 22-24: Full Mix Testing**
- Test model on full mixes (without stem separation)
- Evaluate accuracy degradation
- Decide: is Demucs needed?
- If needed, integrate Demucs
- Re-evaluate with Demucs

**Days 25-26: Post-Processing Refinement**
- Tune onset detection threshold
- Tune minimum interval between hits
- Add per-class thresholds if needed
- Improve MIDI export timing

**Days 27-28: Genre-Specific Testing**
- Test on various genres (rock, electronic, jazz)
- Identify failure modes
- Add genre-specific handling if needed

**Deliverable:** Model tested on real-world uploads

### Week 5-6: Integration & Production

**Days 29-31: Pipeline Integration**
- Build end-to-end inference pipeline
- Add caching system (avoid reprocessing)
- Optimize inference speed (batch processing, GPU utilization)
- Add progress indicators

**Days 32-35: Game Integration**
- Export MIDI in your game's format
- Test in game engine
- Adjust timing/quantization if needed
- Handle edge cases (short songs, long songs, silence)

**Days 36-40: Polish & Testing**
- Test with 100+ real user uploads
- Fix bugs
- Optimize memory usage
- Add error handling
- Write documentation

**Days 41-42: User Testing**
- Internal testing with game
- Collect feedback
- Final adjustments

**Deliverable:** Production-ready transcription system

---

## 10. Expected Performance & Metrics

### Target Metrics (E-GMD Test Set)

Based on SOTA research with E-GMD:

**Overall F-measure:** 70-80%

**Per-Class F-measure:**
- Kick: 85-90% (easiest - distinct frequency)
- Snare: 80-85%
- Hi-hat: 75-80%
- Hi-Tom: 70-75%
- Mid-Tom: 65-75%
- Low-Tom: 65-75%
- Crash: 70-75%
- Ride: 70-75%

**Evaluation Code:**

```python
from mir_eval import transcription

def evaluate_model(predictions, ground_truth, tolerance=0.05):
    """
    Evaluate drum transcription using MIR metrics
    
    Args:
        predictions: List of (time, drum_idx) tuples
        ground_truth: List of (time, drum_idx) tuples
        tolerance: Temporal tolerance (±50ms standard)
    """
    results = {}
    drum_names = ['kick', 'snare', 'hihat', 'hi_tom', 'mid_tom', 'low_tom', 'crash', 'ride']
    
    for drum_idx, drum_name in enumerate(drum_names):
        # Extract onsets for this drum
        pred_times = [p[0] for p in predictions if p[1] == drum_idx]
        true_times = [g[0] for g in ground_truth if g[1] == drum_idx]
        
        # Compute metrics
        if len(true_times) > 0:
            precision, recall, f_measure = transcription.precision_recall_f1_overlap(
                np.array(true_times),
                np.array(pred_times),
                onset_tolerance=tolerance
            )
        else:
            precision = recall = f_measure = 0.0
        
        results[drum_name] = {
            'precision': precision,
            'recall': recall,
            'f_measure': f_measure,
            'num_true': len(true_times),
            'num_pred': len(pred_times)
        }
    
    # Overall metrics
    avg_f = np.mean([r['f_measure'] for r in results.values()])
    
    return results, avg_f
```

---

## 11. Key Advantages of This Revised Approach

### What Makes This Better

1. **No Post-Processing Hacks**
   - Direct 8-class output from E-GMD
   - No heuristics to split toms or cymbals
   - Cleaner, more maintainable

2. **Simpler Training Pipeline**
   - No stem separation during training
   - Faster preprocessing
   - Less complexity

3. **Better Code Quality**
   - PyTorch Lightning eliminates boilerplate
   - Easier to debug
   - Easier to extend

4. **Modern Dependency Management**
   - UV is 10-100x faster than pip
   - Better reproducibility
   - Cleaner project structure

5. **Optional Stem Separation**
   - Test first without Demucs
   - Only add if necessary
   - Saves processing time if not needed

---

## 12. Risks & Mitigation (Updated)

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| E-GMD doesn't generalize to real music | High | Low-Medium | Test early on real uploads; add ADTOF fine-tuning if needed |
| Model needs stem separation | Medium | Medium | Demucs integration is straightforward; test without first |
| UV installation issues on macOS | Low | Medium | Use Homebrew for system deps; UV has good error messages |
| Training takes too long | Low | Low | Use both GPUs; Lightning makes multi-GPU trivial |
| VRAM limitations | Low | Low | Reduce batch size; use gradient accumulation |
| Electronic drums don't match acoustic | Medium | Low | E-GMD has 43 kits with varied timbres; augmentation helps |

---

## 13. Immediate Next Steps (This Week)

### Day 1: Environment Setup

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project
mkdir drum_transcription
cd drum_transcription
uv init --python 3.12

# Configure pyproject.toml (see section 4)
# Then install dependencies
uv sync

# Verify installation
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Day 2: Download E-GMD

```bash
# Download E-GMD (90GB, ~2-4 hours depending on connection)
# From: https://magenta.withgoogle.com/datasets/e-gmd

wget http://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip

# Unzip
unzip e-gmd-v1.0.0.zip -d data/e-gmd/
```

### Day 3: Explore Dataset

```python
# explore_egmd.py
import os
import pretty_midi
import librosa
import matplotlib.pyplot as plt

# Find sample file
audio_file = 'data/e-gmd/drummer1/session1/1_funk_120_beat_4-4.wav'
midi_file = audio_file.replace('.wav', '.mid')

# Load audio
audio, sr = librosa.load(audio_file, sr=22050)
print(f"Audio: {len(audio)} samples, {len(audio)/sr:.2f} seconds")

# Load MIDI
midi = pretty_midi.PrettyMIDI(midi_file)
for instrument in midi.instruments:
    if instrument.is_drum:
        print(f"Drum notes: {len(instrument.notes)}")
        for note in instrument.notes[:5]:
            print(f"  Pitch {note.pitch}: {note.start:.3f}s - {note.end:.3f}s")

# Visualize spectrogram
spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
log_spec = librosa.power_to_db(spec)

plt.figure(figsize=(12, 4))
librosa.display.specshow(log_spec, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-Mel Spectrogram')
plt.tight_layout()
plt.savefig('sample_spectrogram.png')
print("Saved sample_spectrogram.png")
```

### Day 4-7: Implement Preprocessing

See section 2 for feature extraction code. Process entire E-GMD dataset:

```python
# preprocess_egmd.py
import os
import librosa
import numpy as np
import pretty_midi
from tqdm import tqdm
from pathlib import Path

def preprocess_egmd_dataset(egmd_root, output_dir):
    """
    Preprocess entire E-GMD dataset
    Saves spectrograms and labels as .npy files
    """
    audio_files = list(Path(egmd_root).rglob('*.wav'))
    
    for audio_path in tqdm(audio_files):
        # Extract features
        audio, sr = librosa.load(audio_path, sr=22050)
        spec = extract_features(audio)
        
        # Load MIDI labels
        midi_path = str(audio_path).replace('.wav', '.mid')
        labels = midi_to_labels(midi_path, spec.shape[1])
        
        # Save
        output_path = Path(output_dir) / audio_path.relative_to(egmd_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(str(output_path).replace('.wav', '_spec.npy'), spec)
        np.save(str(output_path).replace('.wav', '_labels.npy'), labels)

# Run preprocessing
preprocess_egmd_dataset('data/e-gmd/', 'data/processed/')
```

---

## 14. Resources & Documentation

### Essential Links

**E-GMD Dataset:**
- Download: https://magenta.withgoogle.com/datasets/e-gmd
- Paper: "Learning to Groove with Inverse Sequence Transformations"

**PyTorch Lightning:**
- Docs: https://lightning.ai/docs/pytorch/stable/
- Tutorials: https://lightning.ai/docs/pytorch/stable/starter/introduction.html
- GitHub: https://github.com/Lightning-AI/pytorch-lightning

**UV Package Manager:**
- Docs: https://docs.astral.sh/uv/
- PyTorch Guide: https://docs.astral.sh/uv/guides/integration/pytorch/
- GitHub: https://github.com/astral-sh/uv

**Libraries:**
- Librosa: https://librosa.org/doc/latest/index.html
- Pretty MIDI: https://craffel.github.io/pretty-midi/
- Mido: https://mido.readthedocs.io/
- MIR Eval: https://craffel.github.io/mir_eval/

### Example Projects

- **ADTOF Implementation**: https://github.com/MZehren/ADTOF
- **PyTorch Lightning Examples**: https://github.com/Lightning-AI/pytorch-lightning/tree/master/examples

---

## 15. Future Enhancements (Post-PoC)

1. **Velocity Prediction**
   - Add velocity estimation head to model
   - Enables dynamics for gameplay

2. **Hi-hat State Detection**
   - Distinguish open vs closed hi-hat
   - Adds realism

3. **ADTOF Fine-Tuning**
   - If E-GMD doesn't generalize well
   - Fine-tune on ADTOF's real-world mixes

4. **Tempo/Beat Tracking**
   - Align transcription to song structure
   - Better quantization

5. **User Feedback Loop**
   - Learn from player corrections
   - Improve model over time

6. **Genre-Specific Models**
   - Fine-tune per genre
   - Rock vs Electronic vs Jazz

---

## Conclusion

This revised plan simplifies your workflow significantly:

- **E-GMD gives you perfect 8-class alignment** (no post-processing hacks)
- **PyTorch Lightning eliminates boilerplate** (clean, maintainable code)
- **UV makes dependency management painless** (fast, modern)
- **No stem separation for training** (faster, simpler)
- **Optional Demucs for deployment** (test without it first)

**Key Takeaways:**
1. Start with E-GMD only (444 hours, perfect classes)
2. Use PyTorch Lightning (best for occasional ML work)
3. Use UV for dependencies (10-100x faster than pip)
4. Train without stem separation (simpler pipeline)
5. Test on real uploads before adding Demucs
6. Expect 70-80% F-measure on E-GMD

**Next Actions:**
1. Install UV on all machines
2. Download E-GMD (90GB, start overnight)
3. Set up PyTorch with CUDA via UV
4. Explore E-GMD dataset structure
5. Implement preprocessing pipeline

Good luck! This should be much smoother than the original plan, and you'll have cleaner, more maintainable code with PyTorch Lightning.
