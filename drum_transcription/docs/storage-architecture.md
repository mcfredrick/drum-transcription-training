# Storage Architecture

**Date:** February 7, 2026  
**Decision:** Keep project on SSD, use HDD for large data only

---

## Why This Architecture?

After extracting the 90GB E-GMD dataset and beginning project setup, we had to decide:
- Should the entire project live on the HDD with the dataset?
- Or keep the project on SSD and reference the dataset from HDD?

**We chose: SSD for project, HDD for data**

---

## Rationale

### Benefits of SSD for Project Code
1. **Fast development iteration** - Git operations, code editing, IDE indexing, and module imports are significantly faster on SSD
2. **Better workflow** - The project lives in the natural working directory (`~/Documents/`)
3. **Small footprint** - Project files (configs, Python scripts, notebooks) are tiny (~270KB) and won't grow significantly
4. **Python environment performance** - Package installs/updates and module imports are faster
5. **Development speed** - Code changes, linting, testing, and debugging benefit from SSD speed

### Why HDD is Acceptable for Data
1. **Sequential reads dominate** - Training loads audio files in batches sequentially, which HDDs handle reasonably well
2. **GPU is the bottleneck** - Once data is loaded, GPU computation time far exceeds I/O time
3. **Periodic writes** - Checkpoints and logs are saved occasionally, not continuously
4. **Size requirements** - Dataset (90GB) + processed data (~50-100GB) + checkpoints (~5-10GB) would consume valuable SSD space

---

## Directory Structure

### SSD: `/home/matt/Documents/drum-tranxn/drum_transcription/`
**Purpose:** Active development, code, configurations

```
drum_transcription/
├── .venv/                  # Python virtual environment (UV managed)
├── pyproject.toml          # Project dependencies
├── README.md               # Project documentation
├── docs/                   # Documentation (this file)
│   └── storage-architecture.md
├── configs/                # YAML configuration files
│   ├── test_config.yaml    # Quick test with 20 files
│   └── full_config.yaml    # Full training configuration (to be created)
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                # Training and inference scripts
│   ├── preprocess_egmd.py
│   ├── train.py
│   └── transcribe.py
└── src/                    # Source code modules
    ├── data/               # Data processing
    │   ├── audio_processing.py
    │   ├── midi_processing.py
    │   ├── dataset.py
    │   └── data_module.py
    ├── models/             # Model implementations
    │   └── crnn.py
    └── utils/              # Utility functions
        └── metrics.py
```

**Estimated size:** < 10 MB (excluding .venv which is ~500MB-1GB but benefits from SSD)

### HDD: `/mnt/hdd/drum-tranxn/`
**Purpose:** Large datasets, processed data, training outputs

```
/mnt/hdd/drum-tranxn/
├── e-gmd-v1.0.0/           # Original E-GMD dataset (90GB)
│   ├── drummer1/
│   ├── drummer2/
│   └── ...
├── processed_data/         # Preprocessed spectrograms & labels
│   ├── train/
│   ├── val/
│   └── test/
├── checkpoints/            # Model checkpoints during training
│   ├── test_run/
│   └── full_training/
└── logs/                   # Training logs and tensorboard data
    ├── test_run/
    └── full_training/
```

**Estimated size:** 150-250 GB total

---

## Configuration Pattern

All config files store HDD paths for data and outputs:

```yaml
data:
  egmd_root: "/mnt/hdd/drum-tranxn/e-gmd-v1.0.0"
  processed_dir: "/mnt/hdd/drum-tranxn/processed_data"

training:
  checkpoint_dir: "/mnt/hdd/drum-tranxn/checkpoints"
  log_dir: "/mnt/hdd/drum-tranxn/logs"
```

Code and configs are loaded from the SSD project directory.

---

## Performance Characteristics

### Fast Operations (SSD)
- ✅ Git operations (clone, commit, push, pull)
- ✅ IDE indexing and code completion
- ✅ Module imports (`import src.models.crnn`)
- ✅ Config file reads
- ✅ Code execution and testing
- ✅ Python package installs (`uv add`)

### Acceptable Operations (HDD)
- ⚠️ Loading audio files during training (sequential, batched by DataLoader)
- ⚠️ Reading preprocessed spectrograms (memory-mapped if needed)
- ⚠️ Saving model checkpoints (periodic, not continuous)
- ⚠️ Writing tensorboard logs (buffered)

### Bottleneck (GPU)
- 🚀 Forward/backward passes
- 🚀 Gradient computation
- 🚀 Spectrogram computation (can be GPU-accelerated)

**Result:** Training speed is GPU-bound, not I/O-bound

---

## Working with This Setup

### Starting a Development Session
```bash
cd ~/Documents/drum-tranxn/drum_transcription
uv run python scripts/train.py --config configs/test_config.yaml
```

### All data paths are resolved automatically via configs
- No need to `cd` to HDD locations
- Output paths in configs point to HDD
- Code imports work from SSD

### Version Control
```bash
cd ~/Documents/drum-tranxn/drum_transcription
git add .
git commit -m "Add data preprocessing module"
git push
```

**Note:** HDD data directories are NOT in version control (too large)

---

## Migration from Earlier Setup

An earlier attempt created a duplicate structure at `/mnt/hdd/drum-tranxn/drum_transcription/`.

**Action taken:**
- Kept primary project at `/home/matt/Documents/drum-tranxn/drum_transcription/` (SSD)
- Updated configs to reference HDD for data paths only
- HDD now only contains: dataset, processed_data/, checkpoints/, logs/

**Files to update:**
- `configs/test_config.yaml` - Already points to HDD paths
- New configs should follow the same pattern

---

## Future Considerations

### If SSD Space Becomes Tight
- The `.venv/` directory can be moved to HDD if needed (slower package operations)
- Use `WORKON_HOME` or symlinks to relocate virtual environment

### If Training is I/O Bound
- Preprocess all data to HDF5 or memory-mapped format on HDD
- Use `num_workers > 0` in DataLoader for parallel I/O
- Consider caching preprocessed data in RAM if training set fits (~16-32GB)

### If HDD Space Becomes Tight
- Move old checkpoints to archive storage
- Delete processed data and regenerate when needed (raw data is immutable)
- Use checkpoint compression or save fewer checkpoints

---

## Summary

**This architecture optimizes for:**
- ✅ Development speed and iteration
- ✅ Standard Python workflow
- ✅ Cost-effective storage for large datasets
- ✅ Git-friendly project structure

**Key principle:** Keep hot paths (code, imports, git) on SSD; cold paths (data storage, outputs) on HDD.
