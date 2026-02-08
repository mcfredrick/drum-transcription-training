# Roland TD-17 Drum Transcription Training Guide

This project now uses **ONLY** the Roland TD-17 mapping (26 drum classes). All legacy 8-class mapping code and data has been removed.

## Quick Start

### Option 1: Auto-Training with Best Hyperparameters (Recommended)

The hyperparameter study has already been completed. Best parameters:
- Learning rate: 0.00044547593667580503
- Batch size: 4
- Weight decay: 5.559904341261682e-05
- Optimizer: AdamW

To start training with these parameters:

```bash
# Foreground (interactive)
./auto_train_roland.sh

# Background (with nohup)
./auto_train_roland.sh nohup

# Or manually:
USE_WANDB=false uv run python scripts/train.py --config configs/full_training_config.yaml
```

### Option 2: Run New Hyperparameter Study

If you want to run a new hyperparameter optimization:

```bash
USE_WANDB=false uv run python train_with_optuna.py --n-trials 20 --auto-train
```

## Data Structure

**Roland Preprocessed Data** (26 classes):
- Location: `/mnt/hdd/drum-tranxn/processed_data_roland/`
- Format: HDF5 files with shape `(frames, 26)`
- Splits: `/mnt/hdd/drum-tranxn/processed_data_roland/splits/`

**Legacy Data**: DELETED (moved to `DELETED_legacy_processed_data/`)

## Model Architecture

The model is configured for 26 Roland drum classes:

```yaml
model:
  n_classes: 26
  conv_filters: [32, 64, 128, 256]  # 4 CNN layers
  hidden_size: 256                   # Larger GRU
  num_gru_layers: 3                  # 3 GRU layers
```

## Roland TD-17 Drum Mapping (26 Classes)

| Class | MIDI | Drum Name |
|-------|------|-----------|
| 0 | 36 | Kick |
| 1 | 38 | Snare Head |
| 2 | 37 | Snare X-Stick |
| 3 | 40 | Snare Rim |
| 4 | 48 | Tom 1 Head |
| 5 | 50 | Tom 1 Rim |
| 6 | 45 | Tom 2 Head |
| 7 | 47 | Tom 2 Rim |
| 8 | 43 | Tom 3 Head |
| 9 | 58 | Tom 3 Rim |
| 10 | 41 | Tom 4 Head |
| 11 | 39 | Tom 4 Rim |
| 12 | 42 | Hi-Hat Closed (Bow) |
| 13 | 22 | Hi-Hat Closed (Edge) |
| 14 | 46 | Hi-Hat Open (Bow) |
| 15 | 26 | Hi-Hat Open (Edge) |
| 16 | 44 | Hi-Hat Pedal |
| 17 | 49 | Crash 1 (Bow) |
| 18 | 55 | Crash 1 (Edge) |
| 19 | 57 | Crash 2 (Bow) |
| 20 | 52 | Crash 2 (Edge) |
| 21 | 51 | Ride (Bow) |
| 22 | 59 | Ride (Edge) |
| 23 | 53 | Ride Bell |
| 24 | 54 | Tambourine |
| 25 | 56 | Cowbell |

## Configuration Files

### `configs/full_training_config.yaml`
- **Full training configuration** with optimized hyperparameters
- 150 epochs
- Roland mapping (26 classes)
- Augmentation enabled

### `configs/roland_config.yaml`
- Base Roland configuration
- Can be used as template

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir=/mnt/hdd/drum-tranxn/logs_roland
```

### Check Logs
```bash
tail -f /mnt/hdd/drum-tranxn/logs_roland/training_*.log
```

## Output Locations

- **Checkpoints**: `/mnt/hdd/drum-tranxn/checkpoints_roland/`
- **Logs**: `/mnt/hdd/drum-tranxn/logs_roland/`
- **Best model**: `checkpoints_roland/roland-full-training-{epoch}-{val_loss}.ckpt`

## Preprocessing Data (If Needed)

If you need to reprocess the data:

```bash
uv run python scripts/preprocess_roland.py --config configs/roland_config.yaml
```

## Important Notes

⚠️ **NEVER use legacy mapping** - All legacy code has been removed
⚠️ **Always use 26 classes** - The model expects Roland mapping
⚠️ **Check data paths** - Ensure configs point to `processed_data_roland/`

## Troubleshooting

### "Split file not found"
Make sure Roland data is preprocessed:
```bash
ls /mnt/hdd/drum-tranxn/processed_data_roland/splits/
```

### "Wrong number of classes"
Check config file has `n_classes: 26`

### Training uses wrong data
Verify config paths:
```bash
grep processed_root configs/full_training_config.yaml
# Should output: processed_root: "/mnt/hdd/drum-tranxn/processed_data_roland"
```
