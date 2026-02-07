"""Audio preprocessing utilities for drum transcription."""

import librosa
import numpy as np
from pathlib import Path
from typing import Tuple


def extract_log_mel_spectrogram(
    audio_path: str,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 30.0,
    fmax: float = 11025.0
) -> np.ndarray:
    """
    Extract log-mel spectrogram from audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate
        n_fft: FFT window size
        hop_length: Hop length (stride) for STFT
        n_mels: Number of mel frequency bins
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Log-mel spectrogram of shape (n_mels, time_frames)
    """
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0  # Use power spectrogram
    )
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec


def load_audio(
    audio_path: str,
    sr: int = 22050
) -> Tuple[np.ndarray, int]:
    """
    Load audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate
        
    Returns:
        Tuple of (audio waveform, sample rate)
    """
    audio, sr = librosa.load(audio_path, sr=sr, mono=True)
    return audio, sr


def normalize_audio(audio: np.ndarray, target_db: float = -6.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Audio waveform
        target_db: Target dB level (RMS)
        
    Returns:
        Normalized audio
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    # Calculate target RMS from dB
    target_rms = 10 ** (target_db / 20)
    
    # Scale audio
    if rms > 0:
        audio = audio * (target_rms / rms)
    
    # Clip to prevent overflow
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio


def frame_to_time(frame_idx: int, hop_length: int = 512, sr: int = 22050) -> float:
    """
    Convert frame index to time in seconds.
    
    Args:
        frame_idx: Frame index
        hop_length: Hop length used in STFT
        sr: Sample rate
        
    Returns:
        Time in seconds
    """
    return frame_idx * hop_length / sr


def time_to_frame(time_sec: float, hop_length: int = 512, sr: int = 22050) -> int:
    """
    Convert time in seconds to frame index.
    
    Args:
        time_sec: Time in seconds
        hop_length: Hop length used in STFT
        sr: Sample rate
        
    Returns:
        Frame index
    """
    return int(time_sec * sr / hop_length)


if __name__ == "__main__":
    # Test feature extraction
    import matplotlib.pyplot as plt
    
    # This is a test - replace with actual audio file path
    test_audio = "data/e-gmd/drummer1/session1/1_funk_120_beat_4-4.wav"
    
    if Path(test_audio).exists():
        print(f"Extracting features from: {test_audio}")
        
        spec = extract_log_mel_spectrogram(test_audio)
        print(f"Spectrogram shape: {spec.shape}")
        print(f"Shape format: (n_mels={spec.shape[0]}, time_frames={spec.shape[1]})")
        
        # Visualize
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(
            spec, 
            sr=22050, 
            hop_length=512,
            x_axis='time', 
            y_axis='mel',
            fmin=30,
            fmax=11025
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log-Mel Spectrogram')
        plt.tight_layout()
        plt.savefig('test_spectrogram.png', dpi=150)
        print("Saved visualization to test_spectrogram.png")
    else:
        print(f"Test audio file not found: {test_audio}")
        print("Download E-GMD dataset to test this module.")
