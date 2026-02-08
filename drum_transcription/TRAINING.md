# Training Guide - Roland TD-17 Drum Transcription

Complete guide for training drum transcription models with Roland TD-17 mapping.

## Quick Start

**Prerequisites:** Complete [SETUP.md](SETUP.md) first.

**Start full training:**
```bash
cd ~/Documents/drum-tranxn/drum_transcription
./auto_train_roland.sh
```

This will:
- Use all E-GMD data (~1200+ files)
- Train for 150 epochs with early stopping
- Save checkpoints to `/mnt/hdd/drum-tranxn/checkpoints_roland/`
- Log metrics to TensorBoard
- Target 26 Roland TD-17 drum classes

## Training Configurations

### Three Training Modes

#### 1. Quick Test (5-10 minutes)
**Purpose:** Verify setup is working before long training runs.

```bash
./quick_test.sh
```

**Settings:**
- Files: 10
- Epochs: 5
- Augmentation: Disabled
- Early stopping: Disabled
- Config: `configs/quick_test_config.yaml`

**Use when:**
- Testing code changes
- Verifying model architecture
- Debugging preprocessing issues

---

#### 2. Medium Test (2-3 hours)
**Purpose:** Meaningful validation with realistic training conditions.

```bash
uv run python scripts/train.py --config configs/medium_test_config.yaml
```

**Settings:**
- Files: 100
- Epochs: 20
- Augmentation: Enabled
- Early stopping: Enabled (patience=10)
- Config: `configs/medium_test_config.yaml`

**Use when:**
- Testing hyperparameters
- Validating augmentation strategies
- Prototyping model changes

---

#### 3. Full Training (12-24 hours)
**Purpose:** Production model training with full dataset.

```bash
./auto_train_roland.sh
```

**Settings:**
- Files: All (~1200+)
- Epochs: 150
- Augmentation: Enabled
- Early stopping: Enabled (patience=25)
- Config: `configs/full_training_config.yaml`

**Use when:**
- Training final production models
- Achieving best possible performance
- Creating models for inference

## Hyperparameters

### Current Optimized Settings

The default configs use hyperparameters optimized via Optuna:

```yaml
training:
  num_epochs: 150
  batch_size: 4
  learning_rate: 0.00044547593667580503
  optimizer: AdamW
  weight_decay: 5.559904341261682e-05
  scheduler: "ReduceLROnPlateau"
  scheduler_patience: 5
  scheduler_factor: 0.5
  gradient_clip_val: 1.0
  precision: "16-mixed"

model:
  n_classes: 26  # Roland TD-17
  conv_filters: [32, 64, 128, 256]
  hidden_size: 256
  num_gru_layers: 3
  dropout_cnn: 0.3
  dropout_gru: 0.4
  bidirectional: true
```

### When to Re-optimize

Run hyperparameter optimization (Optuna) when:
- Model architecture changes significantly
- Dataset changes
- Training shows instability
- Starting a new project variant

See [docs/HYPERPARAMETER_OPTIMIZATION.md](docs/HYPERPARAMETER_OPTIMIZATION.md) for details.

## Monitoring Training

### TensorBoard

Launch TensorBoard to monitor training in real-time:

```bash
# Full training logs
tensorboard --logdir=/mnt/hdd/drum-tranxn/logs_roland

# Specific experiment
tensorboard --logdir=/mnt/hdd/drum-tranxn/logs_roland/full_training
```

Open browser to: `http://localhost:6006`

### Key Metrics

**Loss metrics:**
- `train_loss` - Training loss (should decrease)
- `val_loss` - Validation loss (monitor for overfitting)

**Performance metrics:**
- `val_f1` - Validation F1-score (higher is better, target: >0.7)
- `val_precision` - Validation precision
- `val_recall` - Validation recall

**Per-class metrics:**
- `val_f1_kick`, `val_f1_snare_head`, etc. - F1 per drum class
- Identifies which drums are easiest/hardest to transcribe

### Normal Training Behavior

**Healthy training:**
- `train_loss` decreases steadily
- `val_loss` decreases and stabilizes
- `val_f1` increases to 0.6-0.8+
- Learning rate reduces periodically (ReduceLROnPlateau)

**Warning signs:**
- `val_loss` increases while `train_loss` decreases → Overfitting
- Both losses stuck → Learning rate too low or model too small
- NaN values → Learning rate too high or gradient explosion
- Extremely slow convergence → Consider hyperparameter tuning

## Resuming Training

### Automatic Resumption

Training automatically resumes from the last checkpoint if interrupted:

```bash
# Just rerun the same command
./auto_train_roland.sh
```

The trainer will:
- Find the last checkpoint in the checkpoint directory
- Load model weights, optimizer state, and epoch number
- Continue training from where it stopped

### Manual Checkpoint Loading

To resume from a specific checkpoint:

```bash
uv run python scripts/train.py \
    --config configs/full_training_config.yaml \
    --checkpoint /mnt/hdd/drum-tranxn/checkpoints_roland/roland-full-training-epoch=45-val_loss=0.0234.ckpt
```

## Checkpoints

### Checkpoint Strategy

**Saved automatically:**
- Top 3 checkpoints (lowest val_loss)
- Last checkpoint (for resuming)

**Location:**
```
/mnt/hdd/drum-tranxn/checkpoints_roland/
├── full_training/
│   ├── roland-full-training-epoch=78-val_loss=0.0187.ckpt  # Best
│   ├── roland-full-training-epoch=92-val_loss=0.0191.ckpt
│   ├── roland-full-training-epoch=105-val_loss=0.0193.ckpt
│   └── last.ckpt                                           # Latest
```

### Using Checkpoints

**For inference:**
```bash
uv run python scripts/transcribe.py \
    --checkpoint /path/to/best.ckpt \
    --audio /path/to/audio.wav \
    --output output.mid
```

**For evaluation:**
```bash
uv run python scripts/evaluate.py \
    --checkpoint /path/to/best.ckpt \
    --config configs/full_training_config.yaml
```

## Augmentation

### Enabled Augmentations (Full Training)

```yaml
augmentation:
  enabled: true
  time_stretch_prob: 0.3      # Tempo variations
  time_stretch_range: [0.95, 1.05]
  pitch_shift_prob: 0.3       # Slight pitch changes
  pitch_shift_range: [-2, 2]
  volume_scale_prob: 0.5      # Volume variations
  volume_scale_range: [0.8, 1.2]
  reverb_prob: 0.2            # Room acoustics
  noise_prob: 0.3             # Background noise
  noise_range: [0.001, 0.01]
```

### Why Augmentation?

- Improves generalization to new recordings
- Prevents overfitting
- Simulates real-world variations (room acoustics, mic placement, etc.)
- Enables better performance on unseen data

### Disabling Augmentation

For consistent testing or debugging:
```yaml
augmentation:
  enabled: false
```

## Troubleshooting

### "CUDA out of memory"

**Symptoms:** Training crashes with CUDA OOM error.

**Solutions:**
1. Reduce batch size in config:
   ```yaml
   training:
     batch_size: 2  # Reduce from 4
   ```
2. Reduce model size:
   ```yaml
   model:
     conv_filters: [32, 64, 128]  # Remove 256
     hidden_size: 128
   ```
3. Reduce number of workers:
   ```yaml
   hardware:
     num_workers: 2  # Reduce from 4
   ```

### "Loss is NaN"

**Symptoms:** Loss becomes NaN after few steps.

**Solutions:**
1. Reduce learning rate:
   ```yaml
   training:
     learning_rate: 0.0001  # Reduce from 0.00044
   ```
2. Check data preprocessing (invalid values)
3. Enable gradient clipping (already enabled by default)

### "Validation loss not improving"

**Symptoms:** val_loss stuck or increasing.

**Possible causes:**
1. **Overfitting** - Model memorizing training data
   - Increase dropout
   - Enable augmentation
   - Add more training data

2. **Underfitting** - Model too simple
   - Increase model capacity
   - Train more epochs
   - Reduce dropout

3. **Learning rate too low** - Training too slow
   - Increase learning rate
   - Check scheduler settings

### "Training is very slow"

**Solutions:**
1. Verify GPU is being used:
   ```bash
   nvidia-smi
   # Should show python process using GPU
   ```
2. Increase batch size (if memory allows):
   ```yaml
   training:
     batch_size: 8
   ```
3. Increase num_workers:
   ```yaml
   hardware:
     num_workers: 8
   ```
4. Use mixed precision (already enabled):
   ```yaml
   training:
     precision: "16-mixed"
   ```

### "Preprocessing errors"

**Symptoms:** Training fails with data loading errors.

**Solution:** Re-run preprocessing:
```bash
uv run python scripts/preprocess_roland.py \
    --config configs/roland_config.yaml
```

## Advanced Options

### Multi-GPU Training

If you have multiple GPUs:

```yaml
hardware:
  devices: [0, 1]  # Use 2 GPUs
  accelerator: "gpu"
```

### Gradient Accumulation

Simulate larger batch sizes:

```yaml
training:
  batch_size: 2
  accumulate_grad_batches: 4  # Effective batch size = 2 * 4 = 8
```

### Custom Checkpoint Directory

```yaml
checkpoint:
  dirpath: "/custom/path/checkpoints"
  filename: "custom-name-{epoch:02d}-{val_loss:.4f}"
```

## Performance Expectations

### Target Metrics (Full Training)

After 150 epochs with full dataset:
- **Validation F1**: 0.70-0.85
- **Validation Loss**: 0.015-0.025
- **Per-class F1**:
  - Easy (kick, snare): 0.85-0.95
  - Medium (toms, crashes): 0.70-0.85
  - Hard (hi-hat articulations): 0.60-0.75

### Training Time

- **Quick test**: 5-10 minutes (RTX 3070)
- **Medium test**: 2-3 hours (RTX 3070)
- **Full training**: 12-24 hours (RTX 3070)

Times vary based on:
- GPU model
- CPU speed (data loading)
- Disk speed (HDD vs SSD)
- Number of workers

## Next Steps

After training completes:
1. Check TensorBoard metrics
2. Select best checkpoint (lowest val_loss)
3. Test inference - see [INFERENCE.md](INFERENCE.md)
4. Evaluate on test set with `scripts/evaluate.py`

## Additional Resources

- **[SETUP.md](SETUP.md)** - Installation and setup
- **[INFERENCE.md](INFERENCE.md)** - Using trained models
- **[ROLAND_MAPPING.md](ROLAND_MAPPING.md)** - Drum mapping details
- **[docs/HYPERPARAMETER_OPTIMIZATION.md](docs/HYPERPARAMETER_OPTIMIZATION.md)** - Optuna tuning
