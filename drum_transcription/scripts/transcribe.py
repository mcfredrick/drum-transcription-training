"""Inference script for drum transcription."""

import argparse
from pathlib import Path
import sys

import torch
import numpy as np
from scipy.signal import find_peaks

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.crnn import DrumTranscriptionCRNN
from src.data.audio_processing import extract_log_mel_spectrogram
from src.data.midi_processing import create_midi_from_onsets, get_drum_name_mapping, get_drum_names
from src.utils.config import load_config


class DrumTranscriber:
    """End-to-end drum transcription pipeline."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/default_config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize transcriber.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file
            device: Device to run inference on
        """
        self.device = device
        self.config = load_config(config_path)

        # Load model
        print(f"Loading model from: {checkpoint_path}")
        self.model = DrumTranscriptionCRNN.load_from_checkpoint(checkpoint_path)
        self.model.to(device)
        self.model.eval()

        print(f"Model loaded successfully on {device}")

        # Get drum names and mapping
        self.drum_names = get_drum_names()
        self.gm_mapping = get_drum_name_mapping()

    @torch.no_grad()
    def transcribe(
        self,
        audio_path: str,
        output_midi_path: str,
        onset_threshold: float = None,
        min_onset_interval: float = None,
    ):
        """
        Transcribe audio file to MIDI.

        Args:
            audio_path: Path to input audio file
            output_midi_path: Path to save output MIDI file
            onset_threshold: Threshold for onset detection (None = use config)
            min_onset_interval: Minimum interval between onsets in seconds (None = use config)
        """
        print(f"\nTranscribing: {audio_path}")

        # Use config values if not provided
        if onset_threshold is None:
            onset_threshold = self.config.postprocessing.onset_threshold
        if min_onset_interval is None:
            min_onset_interval = self.config.postprocessing.min_onset_interval

        # Extract features
        print("Extracting features...")
        spec = extract_log_mel_spectrogram(
            audio_path,
            sr=self.config.audio.sample_rate,
            n_fft=self.config.audio.n_fft,
            hop_length=self.config.audio.hop_length,
            n_mels=self.config.audio.n_mels,
            fmin=self.config.audio.fmin,
            fmax=self.config.audio.fmax,
        )

        # Convert to tensor
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
        spec_tensor = spec_tensor.to(self.device)

        # Run model
        print("Running inference...")
        predictions = self.model(spec_tensor)  # (1, time, n_classes)
        predictions = predictions[0].cpu().numpy()  # (time, n_classes)

        # Post-process to get onsets
        print("Detecting onsets...")
        onsets = self.postprocess_predictions(
            predictions, threshold=onset_threshold, min_interval=min_onset_interval
        )

        print(f"Detected {len(onsets)} drum hits")

        # Count per drum
        drum_counts = {name: 0 for name in self.drum_names}
        for _, drum_name, _ in onsets:
            drum_counts[drum_name] += 1

        print("\nDetected hits per drum:")
        for drum_name, count in drum_counts.items():
            print(f"  {drum_name:10s}: {count:4d}")

        # Export to MIDI
        print(f"\nExporting to MIDI: {output_midi_path}")
        create_midi_from_onsets(
            onsets=onsets,
            output_path=output_midi_path,
            gm_mapping=self.gm_mapping,
            tempo=120,  # Default tempo
            note_duration=0.1,
        )

        print("Transcription complete!")

        return onsets

    def postprocess_predictions(
        self, predictions: np.ndarray, threshold: float = 0.5, min_interval: float = 0.05
    ) -> list:
        """
        Convert frame-level probabilities to discrete onsets.

        Args:
            predictions: (time, n_classes) array of probabilities
            threshold: Minimum probability for detection
            min_interval: Minimum time between onsets (seconds)

        Returns:
            List of (time, drum_name, velocity) tuples
        """
        onsets = []
        frame_rate = self.config.audio.sample_rate / self.config.audio.hop_length
        min_distance_frames = int(min_interval * frame_rate)

        for class_idx, drum_name in enumerate(self.drum_names):
            # Get predictions for this drum
            class_preds = predictions[:, class_idx]

            # Find peaks above threshold
            peaks, properties = find_peaks(
                class_preds, height=threshold, distance=max(1, min_distance_frames)
            )

            # Convert to onsets
            for peak_idx in peaks:
                onset_time = peak_idx / frame_rate
                # Use prediction probability as proxy for velocity
                velocity = int(min(127, max(1, class_preds[peak_idx] * 127)))

                onsets.append((onset_time, drum_name, velocity))

        # Sort by time
        onsets.sort(key=lambda x: x[0])

        return onsets


def main():
    parser = argparse.ArgumentParser(description="Transcribe drums from audio file")
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    parser.add_argument("output_midi", type=str, help="Path to output MIDI file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--config", type=str, default="configs/default_config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="Onset detection threshold (0-1)"
    )
    parser.add_argument(
        "--min-interval", type=float, default=None, help="Minimum interval between onsets (seconds)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.audio_path).exists():
        print(f"Error: Input file not found: {args.audio_path}")
        return

    # Initialize transcriber
    transcriber = DrumTranscriber(
        checkpoint_path=args.checkpoint, config_path=args.config, device=args.device
    )

    # Transcribe
    transcriber.transcribe(
        audio_path=args.audio_path,
        output_midi_path=args.output_midi,
        onset_threshold=args.threshold,
        min_onset_interval=args.min_interval,
    )


if __name__ == "__main__":
    main()
