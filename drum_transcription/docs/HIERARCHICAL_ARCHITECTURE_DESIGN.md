# Hierarchical Multi-Branch Model Architecture

## The Problem with Current Single-Output Architecture

**Current Model:**
```
Input (spectrogram)
    ↓
CNN + RNN (shared)
    ↓
Single output layer → 11 classes (all at once)
```

**Issues:**
1. **Class confusion**: Model confuses hihat_open/closed, ride/ride_bell
2. **Frequency range mismatch**: Kick (20-100 Hz) vs cymbals (2-8 kHz) share same features
3. **No hierarchy**: Model treats kick vs snare as similar to open vs closed hihat
4. **Imbalanced learning**: All classes weighted equally despite importance differences

## Solution: Hierarchical Multi-Branch Architecture

### Architecture Overview

```
Input (spectrogram)
    ↓
Shared CNN Encoder (frequency features)
    ↓
Shared RNN Encoder (temporal patterns)
    ↓
    ├─→ KICK BRANCH (binary)
    │     └─→ kick/no_kick
    │
    ├─→ SNARE BRANCH (3-class)
    │     └─→ no_snare, snare_head, snare_rim
    │
    ├─→ RHYTHM HAND BRANCH (hierarchical)
    │     ├─→ Primary (4-class): none/hihat/ride/tom
    │     └─→ Variations (conditional):
    │           ├─→ IF hihat: open/closed (2-class)
    │           ├─→ IF ride: bell/body (2-class)
    │           └─→ IF tom: floor/high/mid (3-class)
    │
    └─→ HIHAT PEDAL BRANCH (binary)
          └─→ pedal/no_pedal
```

### Why This Works

**1. Specialized Branches Match Musical Structure**
- Kick and snare are independent rhythmic elements (bass line vs backbeat)
- Rhythm hand is ONE continuous element with variations
- Matches how drummers actually play

**2. Solves Class Confusion**
- Kick branch never sees cymbals → No confusion
- Snare branch only handles snare variants → Focused learning
- Rhythm hand variations are conditional → Only classify open/closed WHEN hihat is detected

**3. Frequency Specialization**
Based on actual drum frequency ranges (see `docs/DRUM_FREQUENCY_REFERENCE.md`):
- Kick branch emphasizes 20-100 Hz (kick drum fundamentals)
- Snare branch emphasizes 200-1000 Hz (shell resonance) + broadband noise (wire rattle)
- Tom branch emphasizes 80-400 Hz (floor tom 80-200 Hz, high toms 100-400 Hz)
- Cymbal branch emphasizes 2-8 kHz (hi-hats, ride, crash)
- Each branch learns features from relevant frequency ranges
- Natural separation: kick (20-100 Hz) has no overlap with cymbals (2-8 kHz)

**4. Prioritized Learning**
- Can weight branch losses based on rhythm game importance
- Kick/snare branches: High weight (critical for gameplay)
- Rhythm hand variations: Lower weight (nice to have)

## Detailed Architecture Design

### 1. Shared Encoder (Bottom)

```python
class SharedEncoder(nn.Module):
    """
    Shared feature extraction for all drum elements.
    Learns general drumming patterns and onset detection.
    """
    def __init__(self):
        # CNN for frequency patterns
        self.cnn = nn.Sequential(
            # Conv blocks (current architecture)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ... more conv layers
        )

        # RNN for temporal patterns
        self.rnn = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        # x: (batch, 1, freq, time)
        features = self.cnn(x)  # (batch, channels, freq', time')

        # Reshape for RNN
        features = features.permute(0, 3, 1, 2)  # (batch, time', channels, freq')
        features = features.flatten(2)  # (batch, time', features)

        # Temporal modeling
        features, _ = self.rnn(features)  # (batch, time', hidden*2)

        return features  # Shared representation
```

### 2. Kick Branch (Simplest)

```python
class KickBranch(nn.Module):
    """
    Binary kick detection.
    Can emphasize low-frequency features.
    """
    def __init__(self, input_size=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Binary output
        )

    def forward(self, shared_features):
        # shared_features: (batch, time, hidden)
        return self.head(shared_features)  # (batch, time, 1)
```

### 3. Snare Branch (Multi-class)

```python
class SnareBranch(nn.Module):
    """
    3-class snare detection: none, head, rim.
    Can emphasize mid-high frequency features.
    """
    def __init__(self, input_size=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # 3 classes: none, head, rim
        )

    def forward(self, shared_features):
        return self.head(shared_features)  # (batch, time, 3)
```

### 4. Rhythm Hand Branch (Hierarchical)

```python
class RhythmHandBranch(nn.Module):
    """
    Hierarchical rhythm hand detection:
    - Primary: Which instrument? (hihat/ride/tom/none)
    - Variations: How is it played? (open/closed, bell/body, etc.)
    """
    def __init__(self, input_size=512):
        super().__init__()

        # Primary instrument detector
        self.primary_head = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)  # none, hihat, ride, tom
        )

        # Variation detectors (conditional)
        self.hihat_variation = nn.Linear(256, 2)  # open/closed
        self.ride_variation = nn.Linear(256, 2)   # bell/body
        self.tom_variation = nn.Linear(256, 3)    # floor/high/mid

        # Shared feature layer for variations
        self.variation_features = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, shared_features):
        # Primary detection
        primary = self.primary_head(shared_features)  # (batch, time, 4)

        # Variation detection (always computed, masked during loss)
        var_features = self.variation_features(shared_features)
        hihat_var = self.hihat_variation(var_features)  # (batch, time, 2)
        ride_var = self.ride_variation(var_features)    # (batch, time, 2)
        tom_var = self.tom_variation(var_features)      # (batch, time, 3)

        return {
            'primary': primary,
            'hihat_variation': hihat_var,
            'ride_variation': ride_var,
            'tom_variation': tom_var
        }
```

### 5. Hihat Pedal Branch (Binary)

```python
class HihatPedalBranch(nn.Module):
    """
    Binary hihat pedal detection.
    This is hard - might need special low-frequency emphasis.
    """
    def __init__(self, input_size=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, shared_features):
        return self.head(shared_features)  # (batch, time, 1)
```

### 6. Complete Model

```python
class HierarchicalDrumCRNN(pl.LightningModule):
    """
    Complete hierarchical drum transcription model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Shared encoder
        self.encoder = SharedEncoder()
        hidden_size = 512  # bidirectional LSTM output

        # Specialized branches
        self.kick_branch = KickBranch(hidden_size)
        self.snare_branch = SnareBranch(hidden_size)
        self.rhythm_branch = RhythmHandBranch(hidden_size)
        self.pedal_branch = HihatPedalBranch(hidden_size)

        # Branch weights (for loss)
        self.branch_weights = {
            'kick': 1.5,        # High priority
            'snare': 1.5,       # High priority
            'rhythm_primary': 1.0,
            'rhythm_variation': 0.5,  # Lower priority
            'pedal': 0.3        # Lowest priority (hard to detect)
        }

    def forward(self, x):
        # Shared encoding
        features = self.encoder(x)  # (batch, time, hidden)

        # Branch predictions
        kick = self.kick_branch(features)
        snare = self.snare_branch(features)
        rhythm = self.rhythm_branch(features)
        pedal = self.pedal_branch(features)

        return {
            'kick': kick,                  # (batch, time, 1)
            'snare': snare,                # (batch, time, 3)
            'rhythm': rhythm,              # dict with primary + variations
            'pedal': pedal                 # (batch, time, 1)
        }

    def compute_loss(self, predictions, labels):
        """
        Custom hierarchical loss function.
        """
        total_loss = 0.0

        # 1. Kick loss (binary cross-entropy)
        kick_loss = F.binary_cross_entropy_with_logits(
            predictions['kick'].squeeze(-1),
            labels['kick']
        )
        total_loss += self.branch_weights['kick'] * kick_loss

        # 2. Snare loss (multi-class)
        snare_loss = F.cross_entropy(
            predictions['snare'].reshape(-1, 3),
            labels['snare'].reshape(-1).long()
        )
        total_loss += self.branch_weights['snare'] * snare_loss

        # 3. Rhythm hand primary loss
        rhythm = predictions['rhythm']
        primary_loss = F.cross_entropy(
            rhythm['primary'].reshape(-1, 4),
            labels['rhythm_primary'].reshape(-1).long()
        )
        total_loss += self.branch_weights['rhythm_primary'] * primary_loss

        # 4. Rhythm hand variation losses (conditional)
        # Only compute loss where that instrument is active

        # Hihat variation (only where hihat is primary)
        hihat_mask = (labels['rhythm_primary'] == 1)  # 1 = hihat
        if hihat_mask.sum() > 0:
            hihat_var_loss = F.cross_entropy(
                rhythm['hihat_variation'][hihat_mask],
                labels['hihat_variation'][hihat_mask].long()
            )
            total_loss += self.branch_weights['rhythm_variation'] * hihat_var_loss

        # Ride variation (only where ride is primary)
        ride_mask = (labels['rhythm_primary'] == 2)  # 2 = ride
        if ride_mask.sum() > 0:
            ride_var_loss = F.cross_entropy(
                rhythm['ride_variation'][ride_mask],
                labels['ride_variation'][ride_mask].long()
            )
            total_loss += self.branch_weights['rhythm_variation'] * ride_var_loss

        # Tom variation (only where tom is primary)
        tom_mask = (labels['rhythm_primary'] == 3)  # 3 = tom
        if tom_mask.sum() > 0:
            tom_var_loss = F.cross_entropy(
                rhythm['tom_variation'][tom_mask],
                labels['tom_variation'][tom_mask].long()
            )
            total_loss += self.branch_weights['rhythm_variation'] * tom_var_loss

        # 5. Hihat pedal loss
        pedal_loss = F.binary_cross_entropy_with_logits(
            predictions['pedal'].squeeze(-1),
            labels['pedal']
        )
        total_loss += self.branch_weights['pedal'] * pedal_loss

        return total_loss
```

## Label Preprocessing

Current labels need to be restructured:

```python
def convert_labels_to_hierarchical(original_labels):
    """
    Convert 11-class labels to hierarchical format.

    Input: (batch, time, 11) - one-hot for each class
    Output: dict with hierarchical labels
    """
    batch, time, _ = original_labels.shape

    hierarchical_labels = {
        'kick': original_labels[:, :, 0],  # Class 0

        'snare': torch.zeros(batch, time, dtype=torch.long),
        # 0 = none, 1 = snare_head (class 1), 2 = snare_rim (class 2)

        'rhythm_primary': torch.zeros(batch, time, dtype=torch.long),
        # 0 = none, 1 = hihat, 2 = ride, 3 = tom

        'hihat_variation': torch.zeros(batch, time, dtype=torch.long),
        # 0 = closed (class 5), 1 = open (class 6)

        'ride_variation': torch.zeros(batch, time, dtype=torch.long),
        # 0 = body (class 9), 1 = bell (class 10)

        'tom_variation': torch.zeros(batch, time, dtype=torch.long),
        # 0 = floor (class 7), 1 = high (class 8), 2 = mid

        'pedal': original_labels[:, :, 4]  # Class 4
    }

    # Fill in snare labels
    snare_head_mask = original_labels[:, :, 1] == 1
    snare_rim_mask = original_labels[:, :, 2] == 1
    hierarchical_labels['snare'][snare_head_mask] = 1
    hierarchical_labels['snare'][snare_rim_mask] = 2

    # Fill in rhythm hand primary
    hihat_mask = (original_labels[:, :, 5] == 1) | (original_labels[:, :, 6] == 1)
    ride_mask = (original_labels[:, :, 9] == 1) | (original_labels[:, :, 10] == 1)
    tom_mask = (original_labels[:, :, 7] == 1) | (original_labels[:, :, 8] == 1)

    hierarchical_labels['rhythm_primary'][hihat_mask] = 1
    hierarchical_labels['rhythm_primary'][ride_mask] = 2
    hierarchical_labels['rhythm_primary'][tom_mask] = 3

    # Fill in hihat variation (only where hihat is primary)
    hihat_open_mask = original_labels[:, :, 6] == 1
    hierarchical_labels['hihat_variation'][hihat_mask] = 0  # Default closed
    hierarchical_labels['hihat_variation'][hihat_open_mask] = 1

    # Fill in ride variation (only where ride is primary)
    ride_bell_mask = original_labels[:, :, 10] == 1
    hierarchical_labels['ride_variation'][ride_mask] = 0  # Default body
    hierarchical_labels['ride_variation'][ride_bell_mask] = 1

    # Fill in tom variation
    # (depends on your tom class structure)

    return hierarchical_labels
```

## Training Strategy

### Phase 1: Train with Current Architecture (Baseline)
Keep your current model as baseline for comparison.

### Phase 2: Train Hierarchical Model
```python
# New training script
python scripts/train_hierarchical.py --config configs/hierarchical_config.yaml
```

### Phase 3: Compare Performance
```
Metric                  Current    Hierarchical    Improvement
----------------------------------------------------------------
Kick (Recall)           39.7%      ???             +?%
Snare (Recall)          42.6%      ???             +?%
Rhythm Hand (Recall)    76.2%      ???             +?%
Hihat Open (Precision)  27.5%      ???             +?%
Ride Bell (Precision)   23.3%      ???             +?%
```

## Expected Benefits

### 1. **Higher Recall on Kick/Snare**
- Dedicated branches with focused learning
- Can weight loss higher for these branches
- Expected: 40% → 70-80% recall

### 2. **Better Precision on Variations**
- Conditional learning (only classify variation when instrument detected)
- Hierarchical structure matches the problem
- Expected: 27% → 60-70% precision on hihat_open

### 3. **Faster Convergence**
- Simpler per-branch problems
- Less interference between unrelated classes
- Expected: 20-30% fewer epochs to converge

### 4. **Better Interpretability**
- Can analyze each branch separately
- Easier to debug (which branch is failing?)
- Can disable/tune branches independently

## Comparison: Hierarchical vs Separate Models

### Option A: Hierarchical (Recommended)
```
Single model with multiple branches
Shared encoder, specialized heads
```
**Pros:**
- Efficient (single forward pass)
- Shared low-level features
- End-to-end training
- Easy deployment

**Cons:**
- More complex architecture
- Custom loss function needed

### Option B: Separate Models
```
4 independent models:
- Kick detector
- Snare detector
- Cymbal classifier
- Tom classifier
```
**Pros:**
- Maximally specialized
- Can use different architectures
- Easy to understand
- Can train/tune independently

**Cons:**
- 4x inference time (or need parallelization)
- 4x training time
- No shared learning
- Complex deployment (4 models)
- Harder to handle simultaneous hits

### Recommendation: Hierarchical

**Why:**
- Single forward pass (fast inference for real-time gameplay)
- Shared encoder learns general drum features
- Still gets specialization through branches
- Easier to deploy (one model file)

**When to use Separate Models:**
- If branches don't help (but try hierarchical first)
- If you need extreme specialization
- If you have unlimited compute budget

## Implementation Plan

### Week 1: Architecture Implementation
- [ ] Implement `SharedEncoder`
- [ ] Implement branch modules
- [ ] Implement `HierarchicalDrumCRNN`
- [ ] Implement custom loss function
- [ ] Write label conversion code

### Week 2: Training & Validation
- [ ] Create hierarchical config
- [ ] Train on small subset (sanity check)
- [ ] Train full model
- [ ] Compare with baseline

### Week 3: Evaluation & Refinement
- [ ] Evaluate on validation set
- [ ] Tune branch weights
- [ ] Optimize thresholds per branch
- [ ] Test in rhythm game

## Next Steps

Want me to:

1. **Implement the hierarchical architecture?**
   - Create `src/models/hierarchical_crnn.py`
   - Create label conversion utilities
   - Create training script

2. **Run per-class threshold analysis first?**
   - See how much we can improve current model
   - Then decide if hierarchical retraining is worth it

3. **Both in parallel?**
   - You run threshold analysis
   - I implement hierarchical architecture
   - Compare results

**What would you prefer?**
