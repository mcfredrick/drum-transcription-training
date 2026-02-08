#!/usr/bin/env python3
"""Training script with Optuna hyperparameter optimization."""

import argparse
from pathlib import Path
import sys
import os
from typing import Dict, Any

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.models.crnn import DrumTranscriptionCRNN
from src.data.data_module import EGMDDataModule
from src.utils.config import load_config


def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna optimization."""
    
    # Suggest hyperparameters (aligned with your target configs)
    lr = trial.suggest_float('lr', 1e-4, 2e-3, log=True)  # 0.0001 to 0.002 (covers 0.0008 and 0.001)
    batch_size = trial.suggest_categorical('batch_size', [4, 6, 8])  # Smaller batch sizes for memory
    weight_decay = trial.suggest_float('weight_decay', 5e-5, 2e-4, log=True)  # 0.00005 to 0.0002 (covers 0.0001)
    optimizer = trial.suggest_categorical('optimizer', ['AdamW'])  # Focus on AdamW (your target)
    
    # Load base config
    config = load_config('configs/roland_config.yaml')
    
    # Override with trial parameters
    config.training.learning_rate = lr
    config.training.batch_size = batch_size
    config.training.weight_decay = weight_decay
    config.training.optimizer = optimizer
    
    # Update model config for 26 classes
    config.model.n_classes = 26
    
    print(f"\nTrial {trial.number}: lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}, optimizer={optimizer}")
    
    # Create data module
    data_module = EGMDDataModule(
        processed_root=config.data.processed_root,
        splits_dir=config.data.splits_dir,
        batch_size=batch_size,
        num_workers=config.hardware.num_workers,
        use_hdf5=True,
        augmentation_config=config.augmentation.to_dict() if hasattr(config.augmentation, 'to_dict') else None
    )
    
    # Create model
    model = DrumTranscriptionCRNN(
        n_mels=config.model.n_mels,
        n_classes=config.model.n_classes,
        conv_filters=config.model.conv_filters,
        conv_kernel_size=config.model.conv_kernel_size,
        pool_size=config.model.pool_size,
        dropout_cnn=config.model.dropout_cnn,
        hidden_size=config.model.hidden_size,
        num_gru_layers=config.model.num_gru_layers,
        dropout_gru=config.model.dropout_gru,
        bidirectional=config.model.bidirectional,
        learning_rate=lr,
        weight_decay=weight_decay,
        scheduler_patience=config.training.scheduler_patience,
        scheduler_factor=config.training.scheduler_factor,
        class_weights=config.loss.class_weights
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        dirpath=f'/mnt/hdd/drum-tranxn/optuna_checkpoints/trial_{trial.number}',
        filename=f'trial_{trial.number}-{{epoch:02d}}-{{val_loss:.4f}}'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,  # Shorter patience for quick testing
        min_delta=0.001
    )
    
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_loss')
    
    # Setup logger (check environment variable for W&B)
    use_wandb = os.getenv('USE_WANDB', 'false').lower() == 'true'
    logger = None
    
    if use_wandb:
        try:
            logger = WandbLogger(
                project='drum-transcription-optuna',
                name=f'trial_{trial.number}',
                config={
                    'lr': lr,
                    'batch_size': batch_size,
                    'weight_decay': weight_decay,
                    'optimizer': optimizer,
                    'trial_number': trial.number
                }
            )
        except:
            print(f"W&B not available, using TensorBoard for trial {trial.number}")
            use_wandb = False
    
    if not use_wandb or logger is None:
        # Use TensorBoard logger
        logger = TensorBoardLogger(
            save_dir=f'/mnt/hdd/drum-tranxn/logs_optuna/trial_{trial.number}',
            name=f'trial_{trial.number}'
        )
        print(f"Using TensorBoard for trial {trial.number} (USE_WANDB={os.getenv('USE_WANDB', 'false')})")
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=10,  # Quick testing
        accelerator='gpu',
        devices=[0],
        callbacks=[checkpoint_callback, early_stop_callback, pruning_callback],
        logger=logger,
        precision='16-mixed',
        enable_checkpointing=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    # Train model
    try:
        trainer.fit(model, data_module)
        val_loss = trainer.callback_metrics['val_loss'].item()
        print(f"Trial {trial.number} completed with val_loss: {val_loss:.4f}")
        return val_loss
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('inf')  # Return worst possible value


def main():
    """Main function to run Optuna optimization."""
    parser = argparse.ArgumentParser(description='Optimize drum transcription hyperparameters')
    parser.add_argument(
        '--n-trials',
        type=int,
        default=20,
        help='Number of optimization trials'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        default='drum-transcription-optuna-v2',
        help='Optuna study name'
    )
    parser.add_argument(
        '--storage',
        type=str,
        default='sqlite:///optuna_study.db',
        help='Optuna storage URL'
    )
    parser.add_argument(
        '--auto-train',
        action='store_true',
        help='Automatically start training with best parameters after optimization'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Optuna Hyperparameter Optimization")
    print("="*80)
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Storage: {args.storage}")
    print("="*80)
    
    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='minimize',
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(objective, n_trials=args.n_trials)
    
    # Print results
    print("\n" + "="*80)
    print("Optimization Complete!")
    print("="*80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val_loss: {study.best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save best config
    best_config = {
        'training': {
            'learning_rate': study.best_trial.params['lr'],
            'batch_size': study.best_trial.params['batch_size'],
            'weight_decay': study.best_trial.params['weight_decay'],
            'optimizer': study.best_trial.params['optimizer']
        }
    }
    
    import yaml
    with open('configs/best_optuna_config.yaml', 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print(f"\nBest config saved to: configs/best_optuna_config.yaml")
    
    # Visualize results (optional)
    try:
        import optuna.visualization as vis
        
        # Plot optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html("optuna_history.html")
        
        # Plot parameter importance
        fig = vis.plot_param_importances(study)
        fig.write_html("optuna_importance.html")
        
        print("Visualization plots saved to optuna_history.html and optuna_importance.html")
    except ImportError:
        print("Install optuna[visualization] for plots: pip install optuna[visualization]")
    
    # Auto-train with best parameters if requested
    if args.auto_train:
        print("\n" + "="*80)
        print("Starting automatic training with best parameters...")
        print("="*80)
        
        # Create training command with best parameters
        train_cmd = [
            'uv', 'run', 'python', 'train.py',
            '--config', 'configs/roland_config.yaml',
            '--learning-rate', str(study.best_trial.params['lr']),
            '--batch-size', str(study.best_trial.params['batch_size']),
            '--weight-decay', str(study.best_trial.params['weight_decay']),
            '--optimizer', study.best_trial.params['optimizer']
        ]
        
        print(f"Command: {' '.join(train_cmd)}")
        
        # Start training
        import subprocess
        subprocess.run(train_cmd, check=True)


if __name__ == "__main__":
    main()
