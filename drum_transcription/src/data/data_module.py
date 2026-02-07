"""PyTorch Lightning DataModule for E-GMD dataset."""

import lightning as L
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
import json

from src.data.dataset import EGMDDataset, collate_fn
from src.data.augmentation import DrumAugmentation, NoAugmentation


class EGMDDataModule(L.LightningDataModule):
    """DataModule for E-GMD drum transcription dataset."""
    
    def __init__(
        self,
        processed_root: str,
        splits_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        use_hdf5: bool = True,
        augmentation_config: Optional[dict] = None
    ):
        """
        Initialize DataModule.
        
        Args:
            processed_root: Root directory of preprocessed data
            splits_dir: Directory containing split files
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            use_hdf5: Whether to use HDF5 format
            augmentation_config: Configuration for data augmentation
        """
        super().__init__()
        self.processed_root = processed_root
        self.splits_dir = Path(splits_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_hdf5 = use_hdf5
        
        # Set up augmentation
        if augmentation_config is not None and augmentation_config.get('enabled', False):
            self.train_transform = DrumAugmentation(
                time_stretch_prob=augmentation_config.get('time_stretch_prob', 0.5),
                time_stretch_range=augmentation_config.get('time_stretch_range', [0.9, 1.1]),
                pitch_shift_prob=augmentation_config.get('pitch_shift_prob', 0.5),
                pitch_shift_range=augmentation_config.get('pitch_shift_range', [-2, 2]),
                volume_scale_prob=augmentation_config.get('volume_scale_prob', 0.5),
                volume_scale_range=augmentation_config.get('volume_scale_range', [-6, 6]),
                reverb_prob=augmentation_config.get('reverb_prob', 0.3),
                reverb_delay_range=augmentation_config.get('reverb_delay_range', [1, 3]),
                reverb_attenuation_range=augmentation_config.get('reverb_attenuation_range', [0.2, 0.4]),
                noise_prob=augmentation_config.get('noise_prob', 0.3),
                noise_level_range=augmentation_config.get('noise_level_range', [0.01, 0.05])
            )
        else:
            self.train_transform = NoAugmentation()
        
        self.val_transform = NoAugmentation()
        self.test_transform = NoAugmentation()
        
        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets.
        
        Args:
            stage: 'fit', 'validate', 'test', or None
        """
        # Load split files
        train_files = self._load_split_file('train_split.txt')
        val_files = self._load_split_file('val_split.txt')
        test_files = self._load_split_file('test_split.txt')
        
        if stage == 'fit' or stage is None:
            self.train_dataset = EGMDDataset(
                file_list=train_files,
                processed_root=self.processed_root,
                transform=self.train_transform,
                use_hdf5=self.use_hdf5
            )
            
            self.val_dataset = EGMDDataset(
                file_list=val_files,
                processed_root=self.processed_root,
                transform=self.val_transform,
                use_hdf5=self.use_hdf5
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = EGMDDataset(
                file_list=test_files,
                processed_root=self.processed_root,
                transform=self.test_transform,
                use_hdf5=self.use_hdf5
            )
    
    def _load_split_file(self, filename: str) -> list:
        """Load split file containing list of file paths."""
        split_path = self.splits_dir / filename
        
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        
        with open(split_path, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        
        return files
    
    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )


if __name__ == "__main__":
    # Test DataModule
    print("Testing EGMDDataModule...")
    
    from src.utils.config import load_config
    
    try:
        config = load_config()
        
        data_module = EGMDDataModule(
            processed_root=config.data.processed_root,
            splits_dir=config.data.splits_dir,
            batch_size=config.training.batch_size,
            num_workers=config.hardware.num_workers,
            use_hdf5=False,
            augmentation_config=config.augmentation.to_dict() if hasattr(config, 'augmentation') else None
        )
        
        print("DataModule created successfully!")
        
        # Try to setup (will fail if data not preprocessed)
        try:
            data_module.setup('fit')
            print(f"Train dataset size: {len(data_module.train_dataset)}")
            print(f"Val dataset size: {len(data_module.val_dataset)}")
        except Exception as e:
            print(f"Could not setup datasets (expected if not preprocessed): {e}")
    
    except FileNotFoundError as e:
        print(f"Config file not found: {e}")
        print("Run from project root directory.")
