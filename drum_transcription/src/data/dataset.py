"""PyTorch Dataset for E-GMD drum transcription."""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple
import h5py

from src.data.hierarchical_labels import convert_to_hierarchical


class EGMDDataset(Dataset):
    """
    E-GMD Dataset for drum transcription.
    Loads preprocessed spectrograms and labels.
    """
    
    def __init__(
        self,
        file_list: list,
        processed_root: str,
        transform: Optional[Callable] = None,
        use_hdf5: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            file_list: List of file paths (relative to E-GMD root)
            processed_root: Root directory of preprocessed data
            transform: Optional data augmentation transform
            use_hdf5: Whether data is stored in HDF5 format (faster) or .npy
        """
        self.processed_root = Path(processed_root)
        self.transform = transform
        self.use_hdf5 = use_hdf5
        
        # Filter file list to only include files that exist
        self.file_list = self._filter_existing_files(file_list)
        
        # Pre-load HDF5 file handles for faster access
        self.hdf5_handles = {}
        if use_hdf5:
            self._open_hdf5_files()
    
    def _filter_existing_files(self, file_list: list) -> list:
        """Filter file list to only include files that have been successfully preprocessed."""
        existing_files = []
        missing_count = 0
        
        for file_path in file_list:
            if self.use_hdf5:
                processed_path = self.processed_root / file_path.replace('.wav', '.h5')
            else:
                processed_path = self.processed_root / file_path.replace('.wav', '_spec.npy')
            
            if processed_path.exists():
                existing_files.append(file_path)
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"Warning: {missing_count} files in split file do not have preprocessed data. Excluding them from dataset.")
            print(f"Dataset size: {len(existing_files)}/{len(file_list)} files")
        
        return existing_files
    
    def _open_hdf5_files(self):
        """Open HDF5 file handles (for faster loading)."""
        # Find all unique HDF5 files
        hdf5_files = set()
        for file_path in self.file_list:
            hdf5_path = self.processed_root / file_path.replace('.wav', '.h5')
            if hdf5_path.exists():
                hdf5_files.add(str(hdf5_path))
        
        # Open handles
        for hdf5_file in hdf5_files:
            self.hdf5_handles[hdf5_file] = h5py.File(hdf5_file, 'r')
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (spectrogram, labels)
            - spectrogram: (1, n_mels, time)
            - labels: (time, n_classes)
        """
        file_path = self.file_list[idx]
        
        # Load spectrogram and labels
        if self.use_hdf5:
            spec, labels = self._load_hdf5(file_path)
        else:
            spec, labels = self._load_npy(file_path)
        
        # Convert to tensors
        spec = torch.FloatTensor(spec).unsqueeze(0)  # Add channel dimension
        labels = torch.FloatTensor(labels)
        
        # Apply augmentation if provided
        if self.transform is not None:
            spec, labels = self.transform(spec, labels)
        
        return spec, labels
    
    def _load_hdf5(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load from HDF5 file."""
        hdf5_path = self.processed_root / file_path.replace('.wav', '.h5')
        
        if str(hdf5_path) in self.hdf5_handles:
            h5_file = self.hdf5_handles[str(hdf5_path)]
            spec = h5_file['spectrogram'][:]
            labels = h5_file['labels'][:]
        else:
            with h5py.File(hdf5_path, 'r') as h5_file:
                spec = h5_file['spectrogram'][:]
                labels = h5_file['labels'][:]
        
        return spec, labels
    
    def _load_npy(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load from .npy files."""
        spec_path = self.processed_root / file_path.replace('.wav', '_spec.npy')
        labels_path = self.processed_root / file_path.replace('.wav', '_labels.npy')
        
        spec = np.load(spec_path)
        labels = np.load(labels_path)
        
        return spec, labels
    
    def __del__(self):
        """Close HDF5 file handles."""
        for handle in self.hdf5_handles.values():
            handle.close()


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the same length within a batch.
    
    Args:
        batch: List of (spec, labels) tuples
        
    Returns:
        Batched (specs, labels, lengths) tensors
    """
    specs, labels = zip(*batch)
    
    # Find max length in batch
    max_time = max(spec.shape[2] for spec in specs)
    
    # Pad all sequences to max length
    specs_padded = []
    labels_padded = []
    lengths = []
    
    for spec, label in zip(specs, labels):
        time_len = spec.shape[2]
        lengths.append(time_len)
        
        # Pad spectrogram
        if time_len < max_time:
            pad_amount = max_time - time_len
            spec_padded = torch.nn.functional.pad(
                spec, 
                (0, pad_amount), 
                mode='constant', 
                value=spec.min()
            )
        else:
            spec_padded = spec
        specs_padded.append(spec_padded)
        
        # Pad labels
        if time_len < max_time:
            pad_amount = max_time - time_len
            label_padded = torch.nn.functional.pad(
                label, 
                (0, 0, 0, pad_amount), 
                mode='constant', 
                value=0
            )
        else:
            label_padded = label
        labels_padded.append(label_padded)
    
    # Stack into batches
    specs_batch = torch.stack(specs_padded)
    labels_batch = torch.stack(labels_padded)
    lengths_batch = torch.LongTensor(lengths)

    return specs_batch, labels_batch, lengths_batch


def hierarchical_collate_fn(batch):
    """
    Custom collate function that converts labels to hierarchical format.

    Args:
        batch: List of (spec, labels) tuples

    Returns:
        Batched (specs, hierarchical_labels, lengths) tensors
        - specs: (batch, 1, n_mels, time)
        - hierarchical_labels: Dict of tensors for each branch
        - lengths: (batch,) - original sequence lengths
    """
    # First use standard collate to pad sequences
    specs_batch, labels_batch, lengths_batch = collate_fn(batch)

    # Convert labels from (batch, time, 12) to hierarchical format
    hierarchical_labels = convert_to_hierarchical(labels_batch)

    return specs_batch, hierarchical_labels, lengths_batch


if __name__ == "__main__":
    # Test dataset
    print("Testing EGMDDataset...")
    
    # This would normally come from your split files
    test_files = [
        "drummer1/session1/1_funk_120_beat_4-4.wav",
        "drummer1/session1/2_latin_140_beat_4-4.wav",
    ]
    
    from augmentation import NoAugmentation
    
    # Create dataset
    dataset = EGMDDataset(
        file_list=test_files,
        processed_root="data/processed",
        transform=NoAugmentation(),
        use_hdf5=False  # Use .npy for testing
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Try loading a sample (will fail if data not preprocessed)
    try:
        spec, labels = dataset[0]
        print(f"Loaded sample 0:")
        print(f"  Spectrogram shape: {spec.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Active frames: {(labels.sum(dim=1) > 0).sum().item()}")
    except Exception as e:
        print(f"Could not load sample (expected if data not preprocessed): {e}")
