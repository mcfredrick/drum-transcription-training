# Drum Transcription Model - Inference API Summary

## Model Overview

A CRNN (Convolutional Recurrent Neural Network) model trained on the E-GMD dataset that transcribes drum audio into MIDI with 8-class output:
- **Classes**: kick, snare, hi-hat, hi-tom, mid-tom, low-tom, crash, ride
- **Input**: Isolated drum audio (WAV, MP3, etc.) - preprocessed with Demucs/Spleeter
- **Output**: General MIDI file (channel 10, standard drum mapping)

## Model Architecture

```
Input: Log-mel spectrogram (1, 128, time_frames)
    ↓
CNN Encoder (3 blocks: 32→64→128 filters)
    ↓
Bidirectional GRU (3 layers, hidden_size=128)
    ↓
Dense layers → Sigmoid
    ↓
Output: Frame-level predictions (time_frames, 8)
```

## Inference Pipeline

### 1. Load Model

```python
import torch
from src.models.crnn import DrumTranscriptionCRNN

# Load trained model
model = DrumTranscriptionCRNN.load_from_checkpoint('path/to/checkpoint.ckpt')
model.eval()
model.to('cuda')  # or 'cpu'
```

### 2. Preprocess Audio

```python
import librosa
import numpy as np

def extract_log_mel_spectrogram(audio_path):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=22050,
        n_fft=2048,
        hop_length=512,      # ~23ms frames (43 FPS)
        n_mels=128,
        fmin=30,             # Captures kick fundamentals
        fmax=11025           # Nyquist
    )
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec  # Shape: (128, time_frames)
```

### 3. Run Inference

```python
@torch.no_grad()
def transcribe_drums(audio_path, model, device='cuda'):
    # Extract features
    spec = extract_log_mel_spectrogram(audio_path)
    
    # Convert to tensor: (1, 1, 128, time)
    spec_tensor = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(device)
    
    # Run model
    predictions = model(spec_tensor)  # (1, time, 8)
    predictions = predictions[0].cpu().numpy()  # (time, 8)
    
    return predictions
```

### 4. Post-process to Onsets

```python
from scipy.signal import find_peaks

def extract_onsets(predictions, threshold=0.5, min_interval=0.05):
    """
    Convert frame-level predictions to discrete drum onsets.
    
    Args:
        predictions: (time, 8) array of probabilities
        threshold: Minimum probability for detection (0-1)
        min_interval: Minimum time between onsets in seconds
    
    Returns:
        List of (time, drum_name, velocity) tuples
    """
    onsets = []
    frame_rate = 22050 / 512  # ~43 FPS
    min_distance_frames = int(min_interval * frame_rate)
    
    drum_names = ['kick', 'snare', 'hihat', 'hi_tom', 'mid_tom', 'low_tom', 'crash', 'ride']
    
    for class_idx, drum_name in enumerate(drum_names):
        # Get predictions for this drum
        class_preds = predictions[:, class_idx]
        
        # Find peaks above threshold
        peaks, _ = find_peaks(
            class_preds,
            height=threshold,
            distance=max(1, min_distance_frames)
        )
        
        # Convert to onsets
        for peak_idx in peaks:
            onset_time = peak_idx / frame_rate
            velocity = int(min(127, max(1, class_preds[peak_idx] * 127)))
            onsets.append((onset_time, drum_name, velocity))
    
    # Sort by time
    onsets.sort(key=lambda x: x[0])
    return onsets
```

### 5. Export to MIDI

```python
import pretty_midi

def create_midi(onsets, output_path, tempo=120):
    """
    Create General MIDI file from drum onsets.
    
    Args:
        onsets: List of (time, drum_name, velocity) tuples
        output_path: Path to save MIDI file
        tempo: BPM (default 120)
    """
    # General MIDI drum mapping (updated for app compatibility)
    GM_DRUM_MAP = {
        'kick': [35, 36],      # Bass Drum 1, Bass Drum 2
        'snare': [37, 38, 40], # Side Stick, Acoustic Snare, Electric Snare
        'hihat': [42, 44, 46], # Closed Hi-Hat, Pedal Hi-Hat, Open Hi-Hat
        'hi_tom': [48, 50],    # Hi-Mid Tom, High Tom
        'mid_tom': [45, 47],   # Low Tom, Low-Mid Tom
        'low_tom': [41, 43],   # Floor Tom (low), Floor Tom (high)
        'crash': [49, 52, 55, 57], # Crash Cymbal 1, China Cymbal, Splash Cymbal, Crash Cymbal 2
        'ride': [51, 53, 59],  # Ride Cymbal 1, Ride Bell, Ride Cymbal 2
    }
    
    # Create MIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # Create drum instrument (channel 9 = channel 10 in GM)
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    
    # Add notes
    for time, drum_name, velocity in onsets:
        note_numbers = GM_DRUM_MAP[drum_name]
        # Use the first note in the list as default (can be made configurable)
        note_number = note_numbers[0]
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=note_number,
            start=time,
            end=time + 0.1  # 100ms duration
        )
        drums.notes.append(note)
    
    # Add to MIDI and save
    midi.instruments.append(drums)
    midi.write(output_path)
```

## Complete API Function

```python
def transcribe_audio_to_midi(
    audio_path: str,
    output_midi_path: str,
    model_checkpoint: str,
    threshold: float = 0.5,
    min_interval: float = 0.05,
    device: str = 'cuda'
) -> dict:
    """
    Complete pipeline: audio file → MIDI file
    
    Args:
        audio_path: Path to input audio (isolated drums)
        output_midi_path: Path to save MIDI file
        model_checkpoint: Path to trained model checkpoint
        threshold: Onset detection threshold (0-1)
        min_interval: Minimum time between onsets (seconds)
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with statistics (number of hits per drum)
    """
    # Load model
    model = DrumTranscriptionCRNN.load_from_checkpoint(model_checkpoint)
    model.eval()
    model.to(device)
    
    # Run inference
    predictions = transcribe_drums(audio_path, model, device)
    
    # Extract onsets
    onsets = extract_onsets(predictions, threshold, min_interval)
    
    # Export to MIDI
    create_midi(onsets, output_midi_path)
    
    # Return statistics
    drum_names = ['kick', 'snare', 'hihat', 'hi_tom', 'mid_tom', 'low_tom', 'crash', 'ride']
    stats = {name: 0 for name in drum_names}
    for _, drum_name, _ in onsets:
        stats[drum_name] += 1
    
    return {
        'total_hits': len(onsets),
        'per_drum': stats,
        'duration': predictions.shape[0] / (22050 / 512)  # seconds
    }
```

## Key Parameters for API

### Model Configuration
- **Sample rate**: 22050 Hz
- **FFT size**: 2048
- **Hop length**: 512 samples (~23ms, 43 FPS)
- **Mel bins**: 128
- **Frequency range**: 30 Hz - 11025 Hz

### Post-processing Defaults
- **Onset threshold**: 0.5 (adjust 0.3-0.7 based on precision/recall needs)
- **Minimum onset interval**: 0.05 seconds (prevents double-triggering)
- **Velocity**: Derived from prediction probability (0-127)

### Performance
- **Inference time**: ~1-2 seconds per minute of audio (GPU)
- **Expected accuracy**: 70-80% F-measure
- **Best for**: Isolated drum tracks (use Demucs/Spleeter first for full mixes)

## Dependencies

```python
torch>=2.4.0
lightning>=2.4.0
librosa>=0.10.2
pretty-midi>=0.2.10
scipy>=1.11.0
numpy>=1.26.0
```

## Example API Usage

```python
# Single file
result = transcribe_audio_to_midi(
    audio_path='drums.wav',
    output_midi_path='output.mid',
    model_checkpoint='checkpoints/best_model.ckpt',
    threshold=0.5,
    device='cuda'
)

print(f"Detected {result['total_hits']} drum hits")
print(f"Per drum: {result['per_drum']}")
```

## Notes for API Implementation

1. **Input validation**: Check audio format, length, sample rate
2. **Error handling**: Handle missing files, CUDA OOM, invalid audio
3. **Batch processing**: Process multiple files in parallel
4. **Caching**: Cache preprocessed spectrograms if re-running with different thresholds
5. **Progress tracking**: Return progress for long audio files
6. **Threshold tuning**: Expose threshold as API parameter for user tuning
7. **File cleanup**: Clean up temporary files after processing

## General MIDI Drum Map (Channel 10) - Updated for App Compatibility

```
Kick: Notes 35, 36 (Bass Drum 1, Bass Drum 2)
Snare: Notes 37, 38, 40 (Side Stick, Acoustic Snare, Electric Snare)
Hi-Hat: Notes 42, 44, 46 (Closed Hi-Hat, Pedal Hi-Hat, Open Hi-Hat)
High Tom: Notes 48, 50 (Hi-Mid Tom, High Tom)
Mid Tom: Notes 45, 47 (Low Tom, Low-Mid Tom)
Floor Tom: Notes 41, 43 (Floor Tom low, Floor Tom high)
Crash: Notes 49, 52, 55, 57 (Crash Cymbal 1, China Cymbal, Splash Cymbal, Crash Cymbal 2)
Ride: Notes 51, 53, 59 (Ride Cymbal 1, Ride Bell, Ride Cymbal 2)
```

All MIDI notes should be on **channel 9** (displayed as channel 10 in most DAWs, as MIDI channels are 0-indexed).

The API will use the first note in each list by default, but can be configured to use any of the alternative notes for variety.
