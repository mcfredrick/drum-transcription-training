#!/bin/bash
# Robust training script for drum transcription
# Handles connection interruptions and automatic resumption

set -e

# Configuration
PROJECT_ROOT="/home/matt/Documents/drum-tranxn/drum_transcription"
CONFIG_FILE="${1:-configs/medium_test_config.yaml}"
LOG_DIR="/mnt/hdd/drum-tranxn/logs"
CHECKPOINT_DIR="/mnt/hdd/drum-tranxn/checkpoints"
LOCK_FILE="/tmp/drum_train.lock"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Check if another training session is running
if [ -f "$LOCK_FILE" ]; then
    pid=$(cat "$LOCK_FILE")
    if ps -p "$pid" > /dev/null 2>&1; then
        error "Another training session is already running (PID: $pid)"
        echo "If you're sure no other training is running, remove: $LOCK_FILE"
        exit 1
    else
        warning "Stale lock file found. Removing..."
        rm -f "$LOCK_FILE"
    fi
fi

# Create lock file
echo $$ > "$LOCK_FILE"

# Cleanup function
cleanup() {
    log "Cleaning up..."
    rm -f "$LOCK_FILE"
}

# Trap for cleanup on exit
trap cleanup EXIT INT TERM

# Change to project directory
cd "$PROJECT_ROOT"

log "==================================================================="
log "Starting robust training session"
log "Config: $CONFIG_FILE"
log "Checkpoint directory: $CHECKPOINT_DIR"
log "Log directory: $LOG_DIR"
log "==================================================================="

# Find the last checkpoint if it exists
LAST_CHECKPOINT="$CHECKPOINT_DIR/medium-test-last.ckpt"
RESUME_FLAG=""

if [ -f "$LAST_CHECKPOINT" ]; then
    log "Found existing checkpoint: $LAST_CHECKPOINT"
    read -p "Resume from last checkpoint? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        RESUME_FLAG="--resume $LAST_CHECKPOINT"
        log "Will resume from checkpoint"
    else
        log "Starting fresh training"
    fi
else
    log "No existing checkpoint found. Starting fresh training."
fi

# Function to run training with retry logic
run_training() {
    local max_retries=3
    local retry_count=0
    local exit_code=0
    
    while [ $retry_count -lt $max_retries ]; do
        log "Training attempt $((retry_count + 1))/$max_retries"
        
        # Run training
        uv run python scripts/train.py \
            --config "$CONFIG_FILE" \
            $RESUME_FLAG \
            2>&1 | tee -a "$LOG_DIR/train_robust.log"
        
        exit_code=${PIPESTATUS[0]}
        
        if [ $exit_code -eq 0 ]; then
            log "Training completed successfully!"
            return 0
        fi
        
        error "Training failed with exit code: $exit_code"
        retry_count=$((retry_count + 1))
        
        if [ $retry_count -lt $max_retries ]; then
            # Check if checkpoint exists for resumption
            if [ -f "$LAST_CHECKPOINT" ]; then
                warning "Will retry from last checkpoint in 10 seconds..."
                RESUME_FLAG="--resume $LAST_CHECKPOINT"
                sleep 10
            else
                error "No checkpoint found. Cannot resume."
                return $exit_code
            fi
        fi
    done
    
    error "Training failed after $max_retries attempts"
    return $exit_code
}

# Run training with retry logic
run_training
final_exit_code=$?

if [ $final_exit_code -eq 0 ]; then
    log "==================================================================="
    log "Training completed successfully!"
    log "==================================================================="
    
    # Show best checkpoint
    if [ -d "$CHECKPOINT_DIR" ]; then
        log "\nSaved checkpoints:"
        ls -lh "$CHECKPOINT_DIR"/medium-test-*.ckpt 2>/dev/null || echo "No checkpoints found"
    fi
    
    # Show TensorBoard command
    log "\nTo view training logs with TensorBoard, run:"
    log "  tensorboard --logdir $LOG_DIR"
else
    error "==================================================================="
    error "Training failed!"
    error "==================================================================="
    error "Check logs at: $LOG_DIR/train_robust.log"
    exit $final_exit_code
fi
