#!/usr/bin/env python3
"""
Robust training runner with automatic resumption and connection error handling.
This script can be run via SSH and will continue training even if the connection drops.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def check_lock_file(lock_file: Path) -> bool:
    """Check if another training session is running."""
    if not lock_file.exists():
        return False
    
    try:
        pid = int(lock_file.read_text().strip())
        # Check if process is still running
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        # Process not running or invalid PID
        lock_file.unlink(missing_ok=True)
        return False


def create_lock_file(lock_file: Path):
    """Create a lock file with the current PID."""
    lock_file.write_text(str(os.getpid()))


def remove_lock_file(lock_file: Path):
    """Remove the lock file."""
    lock_file.unlink(missing_ok=True)


def find_last_checkpoint(checkpoint_dir: Path, prefix: str = "medium-test-last") -> Path | None:
    """Find the last checkpoint file."""
    checkpoint_file = checkpoint_dir / f"{prefix}.ckpt"
    if checkpoint_file.exists():
        return checkpoint_file
    return None


def run_training(config_path: str, max_retries: int = 3, detached: bool = False):
    """
    Run training with automatic resumption on failure.
    
    Args:
        config_path: Path to config file
        max_retries: Maximum number of retry attempts
        detached: Run in detached mode (survives SSH disconnection)
    """
    project_root = Path(__file__).parent
    checkpoint_dir = Path("/mnt/hdd/drum-tranxn/checkpoints")
    log_dir = Path("/mnt/hdd/drum-tranxn/logs")
    lock_file = Path("/tmp/drum_train.lock")
    
    # Check for existing training session
    if check_lock_file(lock_file):
        print("ERROR: Another training session is already running!")
        print(f"If you're sure no other training is running, remove: {lock_file}")
        sys.exit(1)
    
    # Create lock file
    create_lock_file(lock_file)
    
    # Cleanup handler
    def cleanup(signum=None, frame=None):
        print("\nCleaning up...")
        remove_lock_file(lock_file)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        print("=" * 70)
        print("Starting robust training session")
        print(f"Config: {config_path}")
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Log directory: {log_dir}")
        print("=" * 70)
        
        retry_count = 0
        while retry_count < max_retries:
            print(f"\nTraining attempt {retry_count + 1}/{max_retries}")
            
            # Find last checkpoint
            last_checkpoint = find_last_checkpoint(checkpoint_dir)
            resume_args = []
            if last_checkpoint:
                print(f"Found checkpoint: {last_checkpoint}")
                if retry_count == 0:
                    response = input("Resume from last checkpoint? (y/n): ").strip().lower()
                    if response == 'y':
                        resume_args = ["--resume", str(last_checkpoint)]
                    else:
                        print("Starting fresh training")
                else:
                    # Automatically resume on retries
                    resume_args = ["--resume", str(last_checkpoint)]
                    print("Automatically resuming from last checkpoint")
            else:
                print("No existing checkpoint found. Starting fresh training.")
            
            # Build command
            cmd = [
                "uv", "run", "python", "scripts/train.py",
                "--config", config_path,
                *resume_args
            ]
            
            # Create log file
            log_file = log_dir / "train_output.log"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Run training
            print(f"\nRunning: {' '.join(cmd)}")
            print(f"Output will be saved to: {log_file}")
            print("-" * 70)
            
            try:
                with open(log_file, "a") as f:
                    f.write(f"\n{'=' * 70}\n")
                    f.write(f"Training attempt {retry_count + 1}/{max_retries}\n")
                    f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"{'=' * 70}\n\n")
                    f.flush()
                    
                    if detached:
                        # Run detached (survives SSH disconnection)
                        process = subprocess.Popen(
                            cmd,
                            cwd=project_root,
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            start_new_session=True
                        )
                    else:
                        # Run with output to both terminal and file
                        process = subprocess.Popen(
                            cmd,
                            cwd=project_root,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1
                        )
                        
                        # Stream output
                        for line in process.stdout:
                            print(line, end='')
                            f.write(line)
                            f.flush()
                    
                    exit_code = process.wait()
                
                if exit_code == 0:
                    print("\n" + "=" * 70)
                    print("Training completed successfully!")
                    print("=" * 70)
                    
                    # Show best checkpoint
                    if checkpoint_dir.exists():
                        checkpoints = sorted(checkpoint_dir.glob("medium-test-epoch*.ckpt"))
                        if checkpoints:
                            print("\nSaved checkpoints:")
                            for ckpt in checkpoints[:5]:
                                size_mb = ckpt.stat().st_size / (1024 * 1024)
                                print(f"  {ckpt.name} ({size_mb:.1f} MB)")
                    
                    print(f"\nTo view training logs with TensorBoard, run:")
                    print(f"  tensorboard --logdir {log_dir}")
                    
                    cleanup()
                    return
                
                print(f"\nTraining failed with exit code: {exit_code}")
                retry_count += 1
                
                if retry_count < max_retries:
                    # Check if checkpoint exists for resumption
                    if find_last_checkpoint(checkpoint_dir):
                        print(f"Will retry from last checkpoint in 10 seconds...")
                        time.sleep(10)
                    else:
                        print("ERROR: No checkpoint found. Cannot resume.")
                        break
                
            except KeyboardInterrupt:
                print("\n\nTraining interrupted by user.")
                print("Progress has been saved. You can resume later by running this script again.")
                cleanup()
                return
            except Exception as e:
                print(f"\nERROR: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Will retry in 10 seconds...")
                    time.sleep(10)
        
        print("\n" + "=" * 70)
        print(f"Training failed after {max_retries} attempts")
        print("=" * 70)
        print(f"Check logs at: {log_file}")
        
    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Robust training runner with automatic resumption"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/medium_test_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts"
    )
    parser.add_argument(
        "--detached",
        action="store_true",
        help="Run in detached mode (survives SSH disconnection)"
    )
    
    args = parser.parse_args()
    run_training(args.config, args.max_retries, args.detached)


if __name__ == "__main__":
    main()
