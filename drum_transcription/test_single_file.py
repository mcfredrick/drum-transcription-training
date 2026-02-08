#!/usr/bin/env python3
"""Test processing a single file with the fixed Roland script."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scripts.preprocess_roland import process_file
from src.utils.config import load_config

def main():
    print("Testing single file processing with fixed Roland script...")
    
    # Load Roland config
    config = load_config('configs/roland_config.yaml')
    
    # Setup paths
    egmd_root = Path(config.data.egmd_root)
    output_root = Path(config.data.processed_root)
    
    # Get first file from train split
    with open('/mnt/hdd/drum-tranxn/processed_data_roland/splits/train.txt') as f:
        first_file = f.readline().strip()
    
    print(f"Processing file: {first_file}")
    
    # Process the file
    success = process_file(
        file_path=first_file,
        egmd_root=egmd_root,
        output_root=output_root,
        config=config,
        use_hdf5=True,
        force=True
    )
    
    if success:
        print("✅ File processed successfully!")
        
        # Check the output
        output_file = output_root / first_file.replace('.wav', '.h5')
        if output_file.exists():
            import h5py
            import numpy as np
            
            with h5py.File(output_file, 'r') as f:
                spec = f['spectrogram'][:]
                labels = f['labels'][:]
                
            print(f"Output spectrogram shape: {spec.shape}")
            print(f"Output labels shape: {labels.shape}")
            print(f"Number of classes: {labels.shape[1]}")
            
            if labels.shape[1] == 26:
                print("✅ SUCCESS: 26 classes as expected!")
            else:
                print(f"❌ ERROR: Expected 26 classes, got {labels.shape[1]}")
        else:
            print("❌ Output file not found!")
    else:
        print("❌ File processing failed!")

if __name__ == "__main__":
    main()
