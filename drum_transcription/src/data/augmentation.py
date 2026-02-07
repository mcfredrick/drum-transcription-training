"""Data augmentation for drum transcription spectrograms."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DrumAugmentation:
    """Data augmentation for drum spectrograms."""
    
    def __init__(
        self,
        time_stretch_prob: float = 0.5,
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        pitch_shift_prob: float = 0.5,
        pitch_shift_range: Tuple[int, int] = (-2, 2),
        volume_scale_prob: float = 0.5,
        volume_scale_range: Tuple[float, float] = (-6.0, 6.0),
        reverb_prob: float = 0.3,
        reverb_delay_range: Tuple[int, int] = (1, 3),
        reverb_attenuation_range: Tuple[float, float] = (0.2, 0.4),
        noise_prob: float = 0.3,
        noise_level_range: Tuple[float, float] = (0.01, 0.05),
    ):
        """
        Initialize augmentation parameters.
        
        Args:
            time_stretch_prob: Probability of applying time stretching
            time_stretch_range: Range of time stretch factors (min, max)
            pitch_shift_prob: Probability of applying pitch shifting
            pitch_shift_range: Range of pitch shift in semitones (min, max)
            volume_scale_prob: Probability of applying volume scaling
            volume_scale_range: Range of volume scaling in dB (min, max)
            reverb_prob: Probability of applying reverb
            reverb_delay_range: Range of reverb delay in frames (min, max)
            reverb_attenuation_range: Range of reverb attenuation (min, max)
            noise_prob: Probability of adding noise
            noise_level_range: Range of noise level (min, max)
        """
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.volume_scale_prob = volume_scale_prob
        self.volume_scale_range = volume_scale_range
        self.reverb_prob = reverb_prob
        self.reverb_delay_range = reverb_delay_range
        self.reverb_attenuation_range = reverb_attenuation_range
        self.noise_prob = noise_prob
        self.noise_level_range = noise_level_range
    
    def __call__(
        self, 
        spec: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random augmentations to spectrogram and labels.
        
        Args:
            spec: Spectrogram tensor of shape (1, n_mels, time)
            labels: Label tensor of shape (time, n_classes)
            
        Returns:
            Augmented (spec, labels) tuple
        """
        # Time stretching
        if torch.rand(1).item() < self.time_stretch_prob:
            spec, labels = self._time_stretch(spec, labels)
        
        # Pitch shifting (frequency axis)
        if torch.rand(1).item() < self.pitch_shift_prob:
            spec = self._pitch_shift(spec)
        
        # Volume scaling
        if torch.rand(1).item() < self.volume_scale_prob:
            spec = self._volume_scale(spec)
        
        # Reverb
        if torch.rand(1).item() < self.reverb_prob:
            spec = self._apply_reverb(spec)
        
        # Noise
        if torch.rand(1).item() < self.noise_prob:
            spec = self._add_noise(spec)
        
        return spec, labels
    
    def _time_stretch(
        self, 
        spec: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply time stretching to spectrogram and adjust labels.
        
        Args:
            spec: Spectrogram (1, n_mels, time)
            labels: Labels (time, n_classes)
            
        Returns:
            Stretched spec and labels
        """
        # Random stretch factor
        rate = np.random.uniform(*self.time_stretch_range)
        
        # Stretch spectrogram along time axis
        new_length = int(spec.shape[2] / rate)
        spec_stretched = F.interpolate(
            spec.unsqueeze(0),  # (1, 1, n_mels, time)
            size=(spec.shape[1], new_length),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # (1, n_mels, new_time)
        
        # Stretch labels
        labels_stretched = F.interpolate(
            labels.T.unsqueeze(0).unsqueeze(0),  # (1, 1, n_classes, time)
            size=(labels.shape[1], new_length),
            mode='nearest'
        ).squeeze(0).squeeze(0).T  # (new_time, n_classes)
        
        return spec_stretched, labels_stretched
    
    def _pitch_shift(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply pitch shifting by rolling frequency bins.
        
        Args:
            spec: Spectrogram (1, n_mels, time)
            
        Returns:
            Pitch-shifted spectrogram
        """
        # Random pitch shift in bins (approximate semitones)
        shift_bins = np.random.randint(*self.pitch_shift_range)
        
        if shift_bins != 0:
            # Roll along frequency axis
            spec = torch.roll(spec, shifts=shift_bins, dims=1)
            
            # Zero out wrapped regions
            if shift_bins > 0:
                spec[:, :shift_bins, :] = spec.min()
            else:
                spec[:, shift_bins:, :] = spec.min()
        
        return spec
    
    def _volume_scale(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply volume scaling in dB.
        
        Args:
            spec: Spectrogram (1, n_mels, time)
            
        Returns:
            Volume-scaled spectrogram
        """
        # Random volume change in dB
        db_change = np.random.uniform(*self.volume_scale_range)
        
        # Apply scaling (spec is already in dB)
        spec = spec + db_change
        
        return spec
    
    def _apply_reverb(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply simple reverb by adding delayed, attenuated copy.
        
        Args:
            spec: Spectrogram (1, n_mels, time)
            
        Returns:
            Spectrogram with reverb
        """
        # Random delay and attenuation
        delay_frames = np.random.randint(*self.reverb_delay_range)
        attenuation = np.random.uniform(*self.reverb_attenuation_range)
        
        # Create delayed copy
        reverb = F.pad(spec, (delay_frames, 0), mode='constant', value=spec.min())
        reverb = reverb[:, :, :-delay_frames]
        
        # Mix with original
        spec = spec + (reverb * attenuation)
        
        return spec
    
    def _add_noise(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise.
        
        Args:
            spec: Spectrogram (1, n_mels, time)
            
        Returns:
            Noisy spectrogram
        """
        # Random noise level
        noise_level = np.random.uniform(*self.noise_level_range)
        
        # Generate and add noise
        noise = torch.randn_like(spec) * noise_level * spec.std()
        spec = spec + noise
        
        return spec


class NoAugmentation:
    """Dummy augmentation for validation/test (no changes)."""
    
    def __call__(
        self, 
        spec: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return inputs unchanged."""
        return spec, labels


if __name__ == "__main__":
    # Test augmentation
    print("Testing data augmentation...")
    
    # Create dummy spectrogram and labels
    spec = torch.randn(1, 128, 100)  # (1, n_mels, time)
    labels = torch.zeros(100, 8)  # (time, n_classes)
    labels[10, 0] = 1  # Kick at frame 10
    labels[20, 1] = 1  # Snare at frame 20
    
    # Initialize augmentation
    augment = DrumAugmentation()
    
    # Apply augmentation
    spec_aug, labels_aug = augment(spec, labels)
    
    print(f"Original spec shape: {spec.shape}")
    print(f"Augmented spec shape: {spec_aug.shape}")
    print(f"Original labels shape: {labels.shape}")
    print(f"Augmented labels shape: {labels_aug.shape}")
    print(f"Spec value range: [{spec_aug.min():.2f}, {spec_aug.max():.2f}]")
    print(f"Active labels: {labels_aug.sum().item()}")
    print("\nAugmentation test passed!")
