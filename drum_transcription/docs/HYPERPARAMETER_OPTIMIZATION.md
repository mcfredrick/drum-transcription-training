# Hyperparameter Optimization Guide

This guide explains how to use Optuna for hyperparameter optimization with the Roland TD-17 mapping dataset.

## Overview

The project uses Optuna for intelligent hyperparameter search, allowing you to automatically find the best training parameters for your drum transcription model.

## Prerequisites

- Optuna and Optuna integration installed:
  ```bash
  uv add optuna optuna-integration
  ```

- Preprocessed dataset with Roland TD-26 mapping:
  ```bash
  uv run python scripts/preprocess_roland.py --use-hdf5 --num-workers 4
  ```

## Quick Start

### Basic Optimization

```bash
# Run 20 trials with TensorBoard logging (no W&B)
USE_WANDB=false nohup uv run python train_with_optuna.py --n-trials 20 > optuna.log 2>&1 &

# Monitor progress
tail -f optuna.log
```

### With W&B Logging (Optional)

```bash
# Enable W&B for detailed experiment tracking
export USE_WANDB=true
nohup uv run python train_with_optuna.py --n-trials 20 > optuna.log 2>&1 &
```

## Configuration

### Parameter Ranges

The optimization searches over these hyperparameters:

| Parameter | Range | Type | Notes |
|-----------|-------|------|-------|
| `learning_rate` | 1e-4 to 2e-3 (log scale) | float | Covers 0.0008 and 0.001 targets |
| `batch_size` | [8, 12, 16] | categorical | Fits GPU memory constraints |
| `weight_decay` | 5e-5 to 2e-4 (log scale) | float | Around 0.0001 target |
| `optimizer` | `AdamW` | categorical | Optimizer choice |

### Search Strategy

- **Algorithm**: Bayesian optimization (TPE)
- **Objective**: Minimize validation loss
- **Pruning**: Early stopping for bad trials
- **Persistence**: SQLite database for resuming

## Running Optimization

### Command Line Options

```bash
python train_with_optuna.py [OPTIONS]

Options:
  --n-trials N        Number of optimization trials (default: 20)
  --study-name NAME    Study name for organization (default: drum-transcription-optuna)
  --storage URL        Storage URL for persistence (default: sqlite:///optuna_study.db)
```

### Examples

```bash
# Quick test (5 trials)
python train_with_optuna.py --n-trials 5

# Full optimization (50 trials)
python train_with_optuna.py --n-trials 50

# Custom study name
python train_with_optuna.py --study-name "roland-experiment-1"

# Use different storage
python train_with_optuna.py --storage "postgresql://user:pass@localhost/optuna"
```

## Monitoring Progress

### Real-time Logs

```bash
# Follow optimization progress
tail -f optuna.log

# Check current status
ps aux | grep train_with_optuna.py
```

### Study Analysis

```bash
# Check best results
uv run python -c "
import optuna
study = optuna.load_study('drum-transcription-optuna', 'sqlite:///optuna_study.db')
print(f'Trials completed: {len(study.trials)}')
print(f'Best trial: {study.best_trial.number}')
print(f'Best val_loss: {study.best_trial.value:.4f}')
print('Best parameters:')
for k, v in study.best_trial.params.items():
    print(f'  {k}: {v}')
"

# See all trial results
uv run python -c "
import optuna
study = optuna.load_study('drum-transcription-optuna', 'sqlite:///optuna_study.db')
for i, trial in enumerate(study.trials):
    print(f'Trial {trial.number}: val_loss={trial.value:.4f}, params={trial.params}')
"
```

### TensorBoard Visualization

```bash
# Start TensorBoard for all trials
tensorboard --logdir /mnt/hdd/drum-tranxn/logs_optuna --port 6006

# Open http://localhost:6006 in your browser
```

## Results

### Output Files

- `optuna_study.db`: SQLite database with all trial results
- `configs/best_optuna_config.yaml`: Best hyperparameters found
- `optuna_history.html`: Optimization history plot (if optuna[visualization] installed)
- `optuna_importance.html`: Parameter importance plot (if optuna[visualization] installed)
- `logs_optuna/`: TensorBoard logs for each trial

### Best Config Usage

The best configuration is automatically saved to `configs/best_optuna.yaml`. Use it for full training:

```bash
# Train with optimized parameters
uv run python scripts/train.py --config configs/roland_config.yaml \
  learning_rate=$(python -c "
import yaml
with open('configs/best_optuna_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print(config['training']['learning_rate'])
") \
  batch_size=$(python -c "
import yaml
with open('configs/best_optuna_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print(config['training']['batch_size'])
") \
  weight_decay=$(python -c "
import yaml
with open('configs/bot_optuna_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print(config['training']['weight_decay'])
")
```

## Troubleshooting

### Common Issues

**Issue**: `Target size mismatch` error
**Solution**: Ensure dataset was processed with Roland mapping:
```bash
uv run python scripts/preprocess_roland.py --use-hdf5 --num-workers 4
```

**Issue**: W&B login prompts
**Solution**: Disable W&B with environment variable:
```bash
export USE_WANDB=false
```

**Issue**: Study not found
**Solution**: Use same storage path:
```bash
python train_with_optuna.py --storage "sqlite:///optuna_study.db"
```

### Debug Mode

For debugging, run with a single trial:
```bash
python train_with_optuna.py --n-trials 1 --study-name debug
```

## Advanced Usage

### Custom Parameter Ranges

Edit `train_with_optuna.py` to modify search ranges:

```python
# In objective() function:
lr = trial.suggest_float('lr', 5e-5, 5e-3, log=True)  # Wider range
batch_size = trial.suggest_categorical('batch_size', [4, 8, 12, 16, 20])  # More options
```

### Custom Objective Function

Modify the `objective()` function to optimize different metrics:

```python
# Return F1 score instead of validation loss
return trainer.callback_metrics['test_f1']

# Return weighted combination
return 0.7 * val_loss + 0.3 * (1 - test_f1)
```

### Parallel Optimization

Run multiple optimizations simultaneously:
```bash
# Terminal 1: Learning rate focus
python train_with_optuna.py --study-name "lr-optimization" --n-trials 20

# Terminal 2: Batch size focus  
python train_with_optuna.py --study-name "batch-optimization" --n-trials 20
```

## Integration with Training Pipeline

### Automated Workflow

1. **Preprocess dataset** with Roland mapping
2. **Run optimization** to find best parameters
3. **Train final model** with optimized parameters
4. **Evaluate** on test set

### CI/CD Integration

```bash
#!/bin/bash
# Hyperparameter optimization in CI
export USE_WANDB=false
python train_with_optuna.py --n-trials 50 --study-name "ci-optimization"
python scripts/train.py --config configs/best_optuna_config.yaml
```

## Best Practices

1. **Start small**: Begin with 5-10 trials to validate setup
2. **Use appropriate ranges**: Align parameter ranges with target values
3. **Monitor convergence**: Stop if improvement plateaus
4. **Save intermediate results**: Checkpoints for each successful trial
5. **Document experiments**: Keep track of what works and what doesn't

## Environment Variables

| Variable | Default | Description |
|----------|---------|------------|
| `USE_WANDB` | `false` | Enable/disable W&B logging |
| `CUDA_VISIBLE_DEVICES` | `all` | GPU devices to use |
| `WANDB_MODE` | `offline` | W&B sync mode |

## Next Steps

After optimization:

1. **Review best parameters** in `configs/best_optuna_config.yaml`
2. **Update full training config** with optimized values
3. **Run full training** with `configs/roland_config.yaml`
4. **Monitor performance** with TensorBoard or W&B
5. **Evaluate** final model on test set

For detailed results and analysis, see the visualization plots generated by Optuna or check the study database directly.
