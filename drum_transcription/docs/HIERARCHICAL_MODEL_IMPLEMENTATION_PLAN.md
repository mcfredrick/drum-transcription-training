# Hierarchical Drum Transcription Model: Complete Implementation Plan

**Created:** 2026-02-08
**Status:** Planning - Ready for Implementation
**Goal:** Build and train hierarchical multi-branch architecture for rhythm game drum transcription

## Important Notes

- **MIDI Output:** Yes, this model outputs frame-level probabilities that will be converted to MIDI note-on/note-off events
- **Data Source:** Roland TD-17 drum module MIDI data (provides full granularity)
- **Hihat Pedal:** Dropped (low priority, currently 5.8% recall)
- **Crash Cymbals:** **Separate dedicated branch** (accent detector, distinct from rhythm cymbals)
- **Hardware:** NVIDIA RTX 3070 (8GB VRAM - plenty for this architecture)
- **Timeline:** No deadline - optimize for quality over speed

## Architecture Design Decision

**Crash cymbal gets its own branch because:**
- Musical role: Accent/emphasis (infrequent) vs rhythm keeping (continuous)
- Rare events: <1% of frames (class imbalance better handled separately)
- Different optimization: High precision priority (false crashes jarring)
- Distinct characteristics: Explosive onset, very long decay (1-2 sec)
- Simple detection: Binary only (no variations)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Branch Specifications](#branch-specifications)
3. [Implementation Phases](#implementation-phases)
4. [Data Pipeline](#data-pipeline)
5. [Model Architecture Details](#model-architecture-details)
6. [Training Strategy](#training-strategy)
7. [Evaluation Plan](#evaluation-plan)
8. [Success Criteria](#success-criteria)
9. [Risk Mitigation](#risk-mitigation)

---

## Architecture Overview

### Complete Branch Structure

```
Input: Mel Spectrogram (128 bins, 20 Hz - 11 kHz)
    ↓
═══════════════════════════════════════════════════════════════
SHARED ENCODER
═══════════════════════════════════════════════════════════════
    ├─→ CNN Layers (frequency pattern extraction)
    │   - Conv blocks with batch norm and pooling
    │   - Learns: onset patterns, spectral features
    │
    └─→ RNN Layers (temporal pattern modeling)
        - Bidirectional LSTM
        - Learns: rhythmic patterns, temporal context
        - Output: Shared feature vector (batch, time, hidden_size)
═══════════════════════════════════════════════════════════════

        ↓ Shared Features ↓

═══════════════════════════════════════════════════════════════
SPECIALIZED BRANCHES (5 Total)
═══════════════════════════════════════════════════════════════

1. KICK BRANCH
   ├─→ Frequency focus: 20-100 Hz
   ├─→ Task: Binary onset detection
   └─→ Output: kick/no_kick (1 class)

2. SNARE BRANCH
   ├─→ Frequency focus: 200-1000 Hz (shell) + broadband (wires)
   ├─→ Task: Multi-class onset detection & classification
   └─→ Output: no_snare/snare_head/snare_rim (3 classes)

3. TOM BRANCH
   ├─→ Frequency focus: 80-400 Hz
   ├─→ Task: Hierarchical detection
   ├─→ Primary Output: tom/no_tom (onset detection)
   └─→ Variation Output: floor/high/mid (3 classes, conditional)

4. CYMBAL BRANCH (RHYTHM CYMBALS ONLY)
   ├─→ Frequency focus: 2-8 kHz (hihat 2-8 kHz, ride 4-8 kHz)
   ├─→ Task: Hierarchical detection of continuous rhythm cymbals
   ├─→ Primary Output: none/hihat/ride (3 classes)
   └─→ Variation Outputs (conditional):
       ├─→ IF hihat: open/closed (2 classes)
       └─→ IF ride: bell/body (2 classes)

5. CRASH BRANCH (ACCENT DETECTOR)
   ├─→ Frequency focus: 4-8 kHz (broadband, chaotic)
   ├─→ Task: Binary accent detection
   ├─→ Characteristics: Explosive onset, very long decay (1-2 sec)
   ├─→ Frequency: Rare events (~1-5 per 30-sec section)
   └─→ Output: crash/no_crash (1 class, binary)

═══════════════════════════════════════════════════════════════
```

### Why 5 Branches?

**1. Kick (20-100 Hz):**
- Lowest frequency, very distinct
- No overlap with any other drum
- Simple binary detection

**2. Snare (200-1000 Hz + broadband):**
- Unique dual-band signature (shell + wire rattle)
- Head vs rim distinction important for expressiveness
- 3-class output

**3. Tom (80-400 Hz):**
- Distinct from kick (higher) and cymbals (much lower frequency)
- Tonal, resonant character
- Floor/high/mid distinction for fills and patterns
- Hierarchical: Detect tom first, then classify type

**4. Cymbal - Rhythm Only (2-8 kHz):**
- Continuous rhythm keeping cymbals only (hihat and ride)
- Same frequency range, distinguished by temporal characteristics
- Hierarchical: Detect cymbal type first, then variation
- **Hihat:** 2-8 kHz, short decay (closed) or medium decay (open)
- **Ride:** 4-8 kHz, medium decay (~500ms), more tonal than hihat
- Variations: Hihat open/closed, ride bell/body

**5. Crash - Accent Detector (4-8 kHz):**
- **Separate from rhythm cymbals due to distinct role and characteristics**
- Accent/emphasis detector (not continuous rhythm)
- Infrequent: ~1-5 hits per 30-second section (vs 10-30 for ride)
- Explosive onset (sharper attack than ride)
- Very long decay: 1-2 seconds (vs ~500ms for ride)
- Broadband, chaotic spectral content
- No variations (just "crash hit")
- High precision priority (false positives very noticeable/jarring in gameplay)

---

## Branch Specifications

### Branch 1: Kick Detection

**Input:** Shared features (batch, time, hidden_size)
**Output:** Binary probabilities (batch, time, 1)

**Frequency emphasis:** 20-100 Hz
**Training weight:** HIGH (1.5x) - Critical for rhythm game

**Architecture:**
```python
class KickBranch(nn.Module):
    def __init__(self, hidden_size=512):
        self.detector = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Binary output
        )

    def forward(self, shared_features):
        return self.detector(shared_features)  # (batch, time, 1)
```

**Loss function:** Binary Cross-Entropy
**Target recall:** >85%
**Target precision:** >75%

---

### Branch 2: Snare Detection & Classification

**Input:** Shared features (batch, time, hidden_size)
**Output:** 3-class probabilities (batch, time, 3)

**Classes:**
- 0: no_snare
- 1: snare_head
- 2: snare_rim

**Frequency emphasis:**
- Primary: 200-1000 Hz (shell resonance)
- Secondary: 1-10 kHz (wire rattle)

**Training weight:** HIGH (1.5x) - Critical for rhythm game

**Architecture:**
```python
class SnareBranch(nn.Module):
    def __init__(self, hidden_size=512):
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # 3-class output
        )

    def forward(self, shared_features):
        return self.classifier(shared_features)  # (batch, time, 3)
```

**Loss function:** Cross-Entropy
**Target recall:** >85% (combined head+rim)
**Target precision:** >75%

---

### Branch 3: Tom Detection & Classification

**Input:** Shared features (batch, time, hidden_size)
**Outputs:**
- Primary: tom/no_tom (batch, time, 2)
- Variation: floor/high/mid (batch, time, 3)

**Frequency emphasis:** 80-400 Hz
- Floor tom: 80-200 Hz (lower end)
- High/mid toms: 100-400 Hz (upper end)

**Training weight:** MEDIUM (1.0x) - Important for fills/patterns

**Architecture:**
```python
class TomBranch(nn.Module):
    def __init__(self, hidden_size=512):
        # Primary: Is there a tom hit?
        self.primary_detector = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # tom/no_tom
        )

        # Variation: Which tom? (conditional on tom detection)
        self.variation_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # floor/high/mid
        )

    def forward(self, shared_features):
        primary = self.primary_detector(shared_features)
        variation = self.variation_classifier(shared_features)
        return {
            'primary': primary,      # (batch, time, 2)
            'variation': variation   # (batch, time, 3)
        }
```

**Loss function:**
- Primary: Cross-Entropy
- Variation: Cross-Entropy (masked - only compute where tom is detected)

**Target recall:** >75% (tom detection)
**Target precision:** >70%

---

### Branch 4: Rhythm Cymbal Detection & Classification

**Input:** Shared features (batch, time, hidden_size)
**Outputs:**
- Primary: none/hihat/ride (batch, time, 3)
- Hihat variation: open/closed (batch, time, 2)
- Ride variation: bell/body (batch, time, 2)

**Frequency emphasis:**
- Hihat: 2-8 kHz (full range)
- Ride: 4-8 kHz (upper range, more tonal)

**Training weight:**
- Primary: HIGH (1.2x) - Critical for rhythm game
- Variations: MEDIUM (0.5x) - Nice to have

**Rhythm Cymbal Characteristics:**
- **Hihat:** Continuous rhythm, short-medium decay, noise-based
  - Closed: Very short decay (<50ms)
  - Open: Medium decay (~300ms)
- **Ride:** Continuous rhythm, medium decay (~500ms), more tonal
  - Body: Main playing surface
  - Bell: Accent hits, stronger fundamental

**Architecture:**
```python
class CymbalBranch(nn.Module):
    """
    Rhythm cymbal detection (hihat and ride only).
    Crash is handled by separate crash branch.
    """
    def __init__(self, hidden_size=512):
        # Primary: Which rhythm cymbal?
        self.primary_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # none/hihat/ride
        )

        # Hihat variation (conditional on hihat detection)
        self.hihat_variation = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # open/closed
        )

        # Ride variation (conditional on ride detection)
        self.ride_variation = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # bell/body
        )

    def forward(self, shared_features):
        primary = self.primary_classifier(shared_features)
        hihat_var = self.hihat_variation(shared_features)
        ride_var = self.ride_variation(shared_features)
        return {
            'primary': primary,          # (batch, time, 3) - none/hihat/ride
            'hihat_variation': hihat_var,  # (batch, time, 2)
            'ride_variation': ride_var     # (batch, time, 2)
        }
```

**Loss function:**
- Primary: Cross-Entropy (3-class)
- Variations: Cross-Entropy (masked - only compute where hihat/ride detected)

**Target recall:** >80% (hihat/ride detection)
**Target precision (variations):** >70% (open/closed, bell/body)

---

### Branch 5: Crash Cymbal Detection (Accent Detector)

**Input:** Shared features (batch, time, hidden_size)
**Output:** Binary probabilities (batch, time, 1)

**Frequency emphasis:** 4-8 kHz (broadband, chaotic spectral content)
**Training weight:** MEDIUM (0.6x) - Lower priority than rhythm elements

**Crash Characteristics:**
- **Role:** Accent/emphasis, not continuous rhythm
- **Frequency:** Rare (~1-5 per 30-second section)
- **Onset:** Very explosive, sharp attack
- **Decay:** Very long (1-2 seconds) vs ride (~500ms)
- **Spectral:** Broadband chaos, less tonal than ride
- **Gameplay impact:** False positives very noticeable/jarring

**Architecture:**
```python
class CrashBranch(nn.Module):
    """
    Binary crash cymbal detection (accent detector).
    Optimized for high precision (false positives jarring).
    """
    def __init__(self, hidden_size=512):
        self.detector = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Binary output
        )

    def forward(self, shared_features):
        return self.detector(shared_features)  # (batch, time, 1)
```

**Loss function:** Binary Cross-Entropy with class weighting (handle imbalance)
**Target recall:** >75%
**Target precision:** >80% (higher than other branches - avoid false positives)
**Strategy:** Use higher threshold (0.5-0.7) to prioritize precision

---

## Implementation Phases

### Phase 1: Data Pipeline & Label Conversion (Week 1, Days 1-2)

**Goal:** Convert existing 11-class labels to hierarchical format

**Tasks:**

1. **Create label conversion utility:**
   - Input: Current labels (batch, time, 11)
   - Output: Hierarchical labels dict

```python
# File: src/data/hierarchical_labels.py

def convert_to_hierarchical(labels_11class):
    """
    Convert 11-class labels to hierarchical format.

    Input shape: (batch, time, 11)
    Original classes (from Roland TD-17 MIDI):
        0: kick
        1: snare_head
        2: snare_rim
        3: side_stick (can be ignored or mapped to snare_rim)
        4: hihat_pedal (DROPPED - not using)
        5: hihat_closed
        6: hihat_open
        7: floor_tom
        8: high_mid_tom (may need to split into high/mid if data has both)
        9: ride
        10: ride_bell
        (11: crash - if available in data, otherwise derive from metadata)

    Output: dict with keys:
        'kick': (batch, time) - binary
        'snare': (batch, time) - 3 classes (0=none, 1=head, 2=rim)
        'tom_primary': (batch, time) - 2 classes (0=no_tom, 1=tom)
        'tom_variation': (batch, time) - 3 classes (0=floor, 1=high, 2=mid)
        'cymbal_primary': (batch, time) - 3 classes (0=none, 1=hihat, 2=ride)
        'hihat_variation': (batch, time) - 2 classes (0=closed, 1=open)
        'ride_variation': (batch, time) - 2 classes (0=body, 1=bell)
        'crash': (batch, time) - binary
    """
    batch, time, _ = labels_11class.shape

    hierarchical = {}

    # 1. Kick (binary)
    hierarchical['kick'] = labels_11class[:, :, 0].float()

    # 2. Snare (3-class)
    snare = torch.zeros(batch, time, dtype=torch.long)
    snare[labels_11class[:, :, 1] == 1] = 1  # snare_head
    snare[labels_11class[:, :, 2] == 1] = 2  # snare_rim
    hierarchical['snare'] = snare

    # 3. Tom primary (binary: tom/no_tom)
    tom_mask = (labels_11class[:, :, 7] == 1) | (labels_11class[:, :, 8] == 1)
    tom_primary = torch.zeros(batch, time, dtype=torch.long)
    tom_primary[tom_mask] = 1
    hierarchical['tom_primary'] = tom_primary

    # 4. Tom variation (3-class: floor/high/mid)
    tom_variation = torch.zeros(batch, time, dtype=torch.long)
    tom_variation[labels_11class[:, :, 7] == 1] = 0  # floor
    tom_variation[labels_11class[:, :, 8] == 1] = 1  # high/mid
    # Note: If you have separate high/mid in your data, adjust accordingly
    hierarchical['tom_variation'] = tom_variation

    # 5. Cymbal primary (3-class: none/hihat/ride) - RHYTHM CYMBALS ONLY
    cymbal_primary = torch.zeros(batch, time, dtype=torch.long)
    hihat_mask = (labels_11class[:, :, 5] == 1) | (labels_11class[:, :, 6] == 1)
    ride_mask = (labels_11class[:, :, 9] == 1) | (labels_11class[:, :, 10] == 1)
    cymbal_primary[hihat_mask] = 1   # hihat
    cymbal_primary[ride_mask] = 2    # ride
    hierarchical['cymbal_primary'] = cymbal_primary

    # 6. Hihat variation (2-class: closed/open)
    hihat_variation = torch.zeros(batch, time, dtype=torch.long)
    hihat_variation[hihat_mask] = 0  # Default: closed
    hihat_variation[labels_11class[:, :, 6] == 1] = 1  # open
    hierarchical['hihat_variation'] = hihat_variation

    # 7. Ride variation (2-class: body/bell)
    ride_variation = torch.zeros(batch, time, dtype=torch.long)
    ride_variation[ride_mask] = 0  # Default: body
    ride_variation[labels_11class[:, :, 10] == 1] = 1  # bell
    hierarchical['ride_variation'] = ride_variation

    # 8. Crash (binary) - SEPARATE BRANCH
    # If crash is in labels_11class (e.g., index 11 or derive from Roland MIDI)
    if labels_11class.shape[2] > 11:
        hierarchical['crash'] = labels_11class[:, :, 11].float()
    else:
        # Placeholder - will need to extract crash from Roland MIDI data
        hierarchical['crash'] = torch.zeros(batch, time, dtype=torch.float32)

    # Note: Hihat pedal (index 4) is DROPPED - not included in hierarchical labels

    return hierarchical
```

2. **Modify data module:**
   - Update `EGMDDataModule` to return hierarchical labels
   - Create collate function for hierarchical format
   - Test with small batch

3. **Validation:**
   - Verify label counts match original
   - Check no data loss in conversion
   - Visual inspection of converted labels

**Deliverables:**
- `src/data/hierarchical_labels.py`
- Updated `src/data/data_module.py`
- Unit tests for label conversion
- Validation script showing label statistics

---

### Phase 2: Shared Encoder Implementation (Week 1, Days 3-4)

**Goal:** Implement shared CNN+RNN encoder

**Architecture:**

```python
# File: src/models/shared_encoder.py

class SharedEncoder(nn.Module):
    """
    Shared feature encoder for all drum branches.
    Learns general onset patterns and spectral features.
    """
    def __init__(self, config):
        super().__init__()

        # Input: (batch, 1, n_mels, time)
        # n_mels = 128, covering 20 Hz - 11 kHz

        # CNN for frequency pattern extraction
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # Pool in frequency, not time
            # Output: (batch, 32, 64, time)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            # Output: (batch, 64, 32, time)

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            # Output: (batch, 128, 16, time)

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            # Output: (batch, 256, 8, time)
        )

        # Calculate flattened size after CNN
        # 256 channels * 8 frequency bins = 2048 features per time step
        cnn_output_size = 256 * 8

        # RNN for temporal pattern modeling
        self.rnn = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=config['model']['hidden_size'],  # e.g., 256
            num_layers=config['model']['rnn_layers'],    # e.g., 2
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if config['model']['rnn_layers'] > 1 else 0
        )

        # Output size will be hidden_size * 2 (bidirectional)
        self.output_size = config['model']['hidden_size'] * 2

    def forward(self, x):
        """
        Args:
            x: (batch, 1, n_mels, time) - Mel spectrogram

        Returns:
            features: (batch, time', hidden_size*2) - Shared features
        """
        # CNN processing
        features = self.cnn(x)  # (batch, 256, 8, time)

        # Reshape for RNN: (batch, time, features)
        batch, channels, freq, time = features.shape
        features = features.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        features = features.reshape(batch, time, channels * freq)

        # RNN processing
        features, _ = self.rnn(features)  # (batch, time, hidden*2)

        return features
```

**Tasks:**
1. Implement `SharedEncoder` class
2. Test forward pass with dummy data
3. Verify output shapes
4. Count parameters
5. Test on GPU

**Deliverables:**
- `src/models/shared_encoder.py`
- Unit tests
- Parameter count report

---

### Phase 3: Branch Implementation (Week 1, Days 5-7)

**Goal:** Implement all 5 specialized branches

**Tasks:**

1. **Create branch modules:**
   - `src/models/branches/kick_branch.py`
   - `src/models/branches/snare_branch.py`
   - `src/models/branches/tom_branch.py`
   - `src/models/branches/cymbal_branch.py`
   - `src/models/branches/pedal_branch.py`

2. **Test each branch independently:**
   - Forward pass with dummy shared features
   - Verify output shapes
   - Check gradient flow

3. **Create branch factory:**
```python
# src/models/branches/__init__.py

def create_branches(config):
    """Create all branch modules."""
    hidden_size = config['model']['hidden_size'] * 2  # Bidirectional

    branches = {
        'kick': KickBranch(hidden_size),
        'snare': SnareBranch(hidden_size),
        'tom': TomBranch(hidden_size),
        'cymbal': CymbalBranch(hidden_size),
        'pedal': HihatPedalBranch(hidden_size)
    }

    return branches
```

**Deliverables:**
- All 5 branch modules
- Unit tests for each branch
- Integration test with shared encoder

---

### Phase 4: Complete Model Assembly (Week 2, Days 1-2)

**Goal:** Assemble complete hierarchical model

```python
# File: src/models/hierarchical_crnn.py

class HierarchicalDrumCRNN(pl.LightningModule):
    """
    Complete hierarchical drum transcription model.
    5 specialized branches with shared encoder.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # Shared encoder
        self.encoder = SharedEncoder(config)

        # Specialized branches
        hidden_size = self.encoder.output_size
        self.kick_branch = KickBranch(hidden_size)
        self.snare_branch = SnareBranch(hidden_size)
        self.tom_branch = TomBranch(hidden_size)
        self.cymbal_branch = CymbalBranch(hidden_size)
        self.pedal_branch = HihatPedalBranch(hidden_size)

        # Branch weights for loss computation
        self.branch_weights = {
            'kick': 1.5,
            'snare': 1.5,
            'tom_primary': 1.0,
            'tom_variation': 0.7,
            'cymbal_primary': 1.0,
            'cymbal_variation': 0.5,
            'pedal': 0.3
        }

    def forward(self, x):
        """
        Forward pass through all branches.

        Args:
            x: (batch, 1, n_mels, time) - Mel spectrogram

        Returns:
            dict with all branch predictions
        """
        # Shared encoding
        features = self.encoder(x)  # (batch, time', hidden*2)

        # Branch predictions
        predictions = {
            'kick': self.kick_branch(features),
            'snare': self.snare_branch(features),
            'tom': self.tom_branch(features),
            'cymbal': self.cymbal_branch(features),
            'pedal': self.pedal_branch(features)
        }

        return predictions

    def compute_loss(self, predictions, labels, lengths):
        """
        Compute weighted multi-branch loss.
        """
        total_loss = 0.0
        loss_components = {}

        # 1. Kick loss
        kick_loss = self._compute_kick_loss(
            predictions['kick'], labels['kick'], lengths
        )
        total_loss += self.branch_weights['kick'] * kick_loss
        loss_components['kick'] = kick_loss.item()

        # 2. Snare loss
        snare_loss = self._compute_snare_loss(
            predictions['snare'], labels['snare'], lengths
        )
        total_loss += self.branch_weights['snare'] * snare_loss
        loss_components['snare'] = snare_loss.item()

        # 3. Tom losses (primary + variation)
        tom_primary_loss = self._compute_tom_primary_loss(
            predictions['tom']['primary'],
            labels['tom_primary'],
            lengths
        )
        total_loss += self.branch_weights['tom_primary'] * tom_primary_loss
        loss_components['tom_primary'] = tom_primary_loss.item()

        tom_var_loss = self._compute_tom_variation_loss(
            predictions['tom']['variation'],
            labels['tom_variation'],
            labels['tom_primary'],  # Mask: only where tom detected
            lengths
        )
        total_loss += self.branch_weights['tom_variation'] * tom_var_loss
        loss_components['tom_variation'] = tom_var_loss.item()

        # 4. Cymbal losses (primary + variations)
        cymbal_primary_loss = self._compute_cymbal_primary_loss(
            predictions['cymbal']['primary'],
            labels['cymbal_primary'],
            lengths
        )
        total_loss += self.branch_weights['cymbal_primary'] * cymbal_primary_loss
        loss_components['cymbal_primary'] = cymbal_primary_loss.item()

        # Hihat variation (conditional)
        hihat_var_loss = self._compute_hihat_variation_loss(
            predictions['cymbal']['hihat_variation'],
            labels['hihat_variation'],
            labels['cymbal_primary'],  # Mask: only where hihat detected
            lengths
        )
        total_loss += self.branch_weights['cymbal_variation'] * hihat_var_loss
        loss_components['hihat_variation'] = hihat_var_loss.item()

        # Ride variation (conditional)
        ride_var_loss = self._compute_ride_variation_loss(
            predictions['cymbal']['ride_variation'],
            labels['ride_variation'],
            labels['cymbal_primary'],  # Mask: only where ride detected
            lengths
        )
        total_loss += self.branch_weights['cymbal_variation'] * ride_var_loss
        loss_components['ride_variation'] = ride_var_loss.item()

        # 5. Hihat pedal loss
        pedal_loss = self._compute_pedal_loss(
            predictions['pedal'], labels['pedal'], lengths
        )
        total_loss += self.branch_weights['pedal'] * pedal_loss
        loss_components['pedal'] = pedal_loss.item()

        loss_components['total'] = total_loss.item()

        return total_loss, loss_components

    def training_step(self, batch, batch_idx):
        specs, labels_11class, lengths = batch

        # Convert labels to hierarchical format
        labels = convert_to_hierarchical(labels_11class)

        # Forward pass
        predictions = self(specs)

        # Compute loss
        loss, loss_components = self.compute_loss(predictions, labels, lengths)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        for name, value in loss_components.items():
            self.log(f'train_{name}_loss', value)

        return loss

    def validation_step(self, batch, batch_idx):
        specs, labels_11class, lengths = batch
        labels = convert_to_hierarchical(labels_11class)

        predictions = self(specs)
        loss, loss_components = self.compute_loss(predictions, labels, lengths)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        for name, value in loss_components.items():
            self.log(f'val_{name}_loss', value)

        # Compute per-branch metrics (precision, recall, F1)
        metrics = self._compute_validation_metrics(predictions, labels, lengths)
        for name, value in metrics.items():
            self.log(f'val_{name}', value)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
```

**Tasks:**
1. Implement complete model class
2. Implement all loss computation methods
3. Implement metrics computation
4. Test full forward/backward pass
5. Verify gradient flow to all branches

**Deliverables:**
- `src/models/hierarchical_crnn.py`
- Loss computation utilities
- Metrics computation utilities
- Integration tests

---

### Phase 5: Training Pipeline (Week 2, Days 3-4)

**Goal:** Create training script and configuration

**Config file:**
```yaml
# configs/hierarchical_config.yaml

model:
  type: hierarchical_crnn
  hidden_size: 256  # RNN hidden size (will be 512 after bidirectional)
  rnn_layers: 2
  dropout: 0.3

data:
  # Use existing data paths
  processed_dir: /mnt/hdd/drum-tranxn/processed
  batch_size: 16
  num_workers: 4
  # ... other data config

training:
  max_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  gradient_clip_val: 1.0

  # Branch-specific settings
  branch_weights:
    kick: 1.5
    snare: 1.5
    tom_primary: 1.0
    tom_variation: 0.7
    cymbal_primary: 1.0
    cymbal_variation: 0.5
    pedal: 0.3

paths:
  checkpoint_dir: /mnt/hdd/drum-tranxn/checkpoints/hierarchical
  log_dir: /mnt/hdd/drum-tranxn/logs/hierarchical

callbacks:
  early_stopping:
    monitor: val_loss
    patience: 15
    mode: min

  model_checkpoint:
    monitor: val_kick_recall  # Prioritize kick performance
    mode: max
    save_top_k: 3
```

**Training script:**
```python
# scripts/train_hierarchical.py

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def main():
    # Load config
    config = load_config('configs/hierarchical_config.yaml')

    # Setup data
    data_module = EGMDDataModule(config)

    # Create model
    model = HierarchicalDrumCRNN(config)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['paths']['checkpoint_dir'],
        filename='hierarchical-{epoch:02d}-{val_kick_recall:.3f}',
        monitor='val_kick_recall',
        mode='max',
        save_top_k=3
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=pl.loggers.TensorBoardLogger(config['paths']['log_dir']),
        gradient_clip_val=config['training']['gradient_clip_val'],
        accelerator='gpu',
        devices=1
    )

    # Train
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
```

**Deliverables:**
- `configs/hierarchical_config.yaml`
- `scripts/train_hierarchical.py`
- Training documentation

---

### Phase 6: Initial Training & Debugging (Week 2, Days 5-7)

**Goal:** Train on small subset, debug issues

**Tasks:**
1. Train on 10% of data (quick iteration)
2. Monitor all branch losses
3. Check for:
   - Gradient flow issues
   - Exploding/vanishing gradients
   - Imbalanced branch learning
   - NaN losses
4. Adjust hyperparameters if needed
5. Visualize predictions vs ground truth

**Success criteria:**
- All branches learning (loss decreasing)
- No gradient issues
- Reasonable predictions on validation set

**Deliverables:**
- Debug report
- Initial metrics
- Visualization of predictions

---

### Phase 7: Full Training (Week 3)

**Goal:** Train on full dataset

**Training plan:**
- Epochs: 100 (with early stopping)
- Batch size: 16
- Expected time: 12-24 hours on GPU

**Monitoring:**
- TensorBoard for loss curves
- Per-branch metrics (precision, recall, F1)
- Validation metrics every epoch
- Save top 3 checkpoints

**Deliverables:**
- Trained model checkpoints
- Training logs
- Performance curves

---

### Phase 8: Evaluation & Comparison (Week 4)

**Goal:** Compare hierarchical model with baseline

**Evaluation script:**
```python
# scripts/evaluate_hierarchical.py

def evaluate_model(model, dataloader):
    """
    Comprehensive evaluation of hierarchical model.
    """
    results = {
        'kick': {'tp': 0, 'fp': 0, 'fn': 0},
        'snare': {'tp': 0, 'fp': 0, 'fn': 0},
        'tom': {'tp': 0, 'fp': 0, 'fn': 0},
        'cymbal': {'tp': 0, 'fp': 0, 'fn': 0},
        # ... etc
    }

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            predictions = model(batch['specs'])
            # Compute metrics per branch
            # ...

    # Calculate precision, recall, F1 per branch
    metrics = compute_metrics(results)

    return metrics

def compare_with_baseline():
    """
    Compare hierarchical model with baseline single-output model.
    """
    # Load both models
    hierarchical = load_hierarchical_model(checkpoint)
    baseline = load_baseline_model(checkpoint)

    # Evaluate both
    hier_metrics = evaluate_model(hierarchical, val_loader)
    base_metrics = evaluate_model(baseline, val_loader)

    # Generate comparison report
    generate_comparison_table(hier_metrics, base_metrics)
```

**Comparison table:**
```
Class          Baseline       Hierarchical    Improvement
              Rec  Prec  F1   Rec  Prec  F1   ΔRec  ΔPrec  ΔF1
-----------------------------------------------------------------
Kick          40%  81%  53%  ???  ???  ???   +??%  +??%  +??%
Snare         43%  81%  56%  ???  ???  ???   +??%  +??%  +??%
Tom           76%  60%  67%  ???  ???  ???   +??%  +??%  +??%
Hihat         76%  76%  76%  ???  ???  ???   +??%  +??%  +??%
Ride          64%  68%  66%  ???  ???  ???   +??%  +??%  +??%
Hihat Open    93%  28%  42%  ???  ???  ???   +??%  +??%  +??%
Ride Bell     77%  23%  36%  ???  ???  ???   +??%  +??%  +??%
-----------------------------------------------------------------
Overall       51%  68%  59%  ???  ???  ???   +??%  +??%  +??%
```

**Deliverables:**
- Evaluation script
- Comprehensive metrics report
- Comparison with baseline
- Confusion matrices per branch
- Failure case analysis

---

## Training Strategy

### Learning Rate Schedule

**Phase 1: Warm-up (Epochs 1-5)**
- Start: 1e-5
- End: 1e-3
- Linear increase

**Phase 2: Main training (Epochs 6-80)**
- Start: 1e-3
- ReduceLROnPlateau: Factor 0.5, Patience 5

**Phase 3: Fine-tuning (Epochs 81-100)**
- Low LR: ~1e-5
- Focus on precision improvements

### Data Augmentation

```python
# During training
augmentations = {
    'time_stretch': 0.9-1.1,  # ±10% tempo variation
    'pitch_shift': ±2 semitones,  # Small pitch variations
    'time_mask': 5-10 frames,  # Random time masking
    'freq_mask': 5-10 mel bins,  # Random frequency masking
    'mixup': alpha=0.2  # Mix training examples
}
```

### Batch Composition

Ensure each batch has diverse examples:
- Mix of different drum types
- Mix of simple and complex patterns
- Balance positive/negative examples

---

## Evaluation Metrics Strategy

### Threshold-Independent Metrics (Training/Validation) ⭐ RECOMMENDED

**Why use threshold-independent metrics?**

During training, using metrics that don't depend on a fixed threshold helps you understand:
1. **Is the model learning to discriminate?** (High AUC = yes, even if F1@0.5 is low)
2. **Is it just a threshold problem?** (High AUC + low F1 = need better threshold)
3. **Model quality vs threshold optimization** (separate concerns)

**Recommended metrics to log during training:**

```python
# In validation_step():
metrics = {
    # Threshold-independent (PRIMARY during training)
    'roc_auc': ROC-AUC score per branch,
    'pr_auc': Precision-Recall AUC per branch,  # Better for imbalanced (crash)

    # Threshold-dependent (SECONDARY during training)
    'precision_at_0.5': Precision at threshold 0.5,
    'recall_at_0.5': Recall at threshold 0.5,
    'f1_at_0.5': F1 at threshold 0.5,
}
```

### Why AUC Metrics Are Critical

**ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**
- Measures: How well model ranks positives higher than negatives
- Range: 0.5 (random) to 1.0 (perfect)
- **Interpretation:**
  - AUC > 0.85: Excellent model, just need to find right threshold
  - AUC 0.7-0.85: Good model, threshold optimization will help
  - AUC < 0.7: Model quality issue, need better features/architecture

**PR-AUC (Precision-Recall - Area Under Curve):**
- **Better than ROC-AUC for imbalanced classes** (crash, hihat pedal)
- Focuses on positive class performance
- More informative when positive class is rare

**Example interpretation:**

```
Kick Branch:
  ROC-AUC: 0.88  ← Model is good at discriminating kicks
  PR-AUC:  0.82  ← Good performance even accounting for imbalance
  F1@0.5:  0.53  ← Low, but threshold is the problem, not model!

Action: Model is fine, just need to lower threshold to 0.3 or use per-class optimization
```

vs

```
Crash Branch:
  ROC-AUC: 0.62  ← Model struggles to discriminate crashes
  PR-AUC:  0.35  ← Poor performance on rare class
  F1@0.5:  0.28  ← Low

Action: Model quality issue - need better features or more crash examples
```

### Implementation in Model

```python
class HierarchicalDrumCRNN(pl.LightningModule):

    def validation_step(self, batch, batch_idx):
        specs, labels_11class, lengths = batch
        labels = convert_to_hierarchical(labels_11class)
        predictions = self(specs)

        # Compute loss (threshold-dependent)
        loss, loss_components = self.compute_loss(predictions, labels, lengths)

        # Log loss
        self.log('val_loss', loss, prog_bar=True)

        # Store predictions and labels for epoch-end AUC computation
        # (AUC needs all predictions at once)
        self.validation_step_outputs.append({
            'predictions': predictions,
            'labels': labels,
            'lengths': lengths
        })

        return loss

    def on_validation_epoch_end(self):
        """Compute AUC metrics at end of epoch (need all predictions)."""

        # Gather all predictions and labels
        all_preds = ...  # Concatenate from validation_step_outputs
        all_labels = ...

        # Compute AUC metrics per branch
        from sklearn.metrics import roc_auc_score, average_precision_score

        # Kick branch
        kick_probs = torch.sigmoid(all_preds['kick']).cpu().numpy()
        kick_labels = all_labels['kick'].cpu().numpy()
        kick_roc_auc = roc_auc_score(kick_labels, kick_probs)
        kick_pr_auc = average_precision_score(kick_labels, kick_probs)

        self.log('val_kick_roc_auc', kick_roc_auc)  # Threshold-independent
        self.log('val_kick_pr_auc', kick_pr_auc)    # Threshold-independent

        # Also compute F1@0.5 for comparison
        kick_pred_binary = (kick_probs > 0.5).astype(int)
        kick_f1 = f1_score(kick_labels, kick_pred_binary)
        self.log('val_kick_f1_at_0.5', kick_f1)     # Threshold-dependent

        # Repeat for all branches...

        # Clear outputs
        self.validation_step_outputs.clear()
```

### Monitoring During Training

**TensorBoard view:**
```
Epoch 10:
  val_loss: 0.024

  # Kick branch (threshold-independent - watch these!)
  val_kick_roc_auc: 0.87  ← Model is learning well!
  val_kick_pr_auc:  0.81
  val_kick_f1_at_0.5: 0.53  ← Low, but that's OK (threshold issue)

  # Crash branch (imbalanced)
  val_crash_roc_auc: 0.78  ← Good discrimination
  val_crash_pr_auc:  0.65  ← Decent for rare class
  val_crash_f1_at_0.5: 0.32  ← Expected (rare + threshold)
```

**Decision making:**
- **If AUC increasing but F1 flat:** Threshold problem, not model problem
- **If AUC plateauing low:** Model quality issue, need architecture/data changes
- **If AUC high (>0.8) across all branches:** Model is good, ready for threshold optimization

### Branch-Specific AUC Targets

```python
auc_targets = {
    'kick': {
        'roc_auc': 0.85,  # Good discrimination
        'pr_auc': 0.75,   # Balanced class, PR-AUC should be high
    },
    'snare': {
        'roc_auc': 0.85,
        'pr_auc': 0.75,
    },
    'tom': {
        'roc_auc': 0.80,  # Slightly lower (harder problem)
        'pr_auc': 0.70,
    },
    'cymbal_rhythm': {
        'roc_auc': 0.82,
        'pr_auc': 0.72,
    },
    'crash': {
        'roc_auc': 0.75,  # Lower target (rare class, hard problem)
        'pr_auc': 0.55,   # PR-AUC critical for imbalanced
    },
}
```

### When to Stop Training Based on AUC

```
Scenario A: Good model, ready for threshold optimization
  - All branch ROC-AUC > 0.80
  - Stop training, proceed to threshold optimization

Scenario B: Some branches struggling
  - Kick/snare ROC-AUC > 0.85 ✓
  - Crash ROC-AUC < 0.65 ✗
  - Continue training, possibly adjust crash branch weight or add augmentation

Scenario C: Model not learning
  - Most branches ROC-AUC < 0.70 after 30 epochs
  - Stop, diagnose issue (architecture, data, preprocessing)
```

---

## Evaluation Plan

### Quantitative Metrics

**Per-branch metrics:**
1. Precision
2. Recall
3. F1-score
4. Support (number of positive examples)

**Overall metrics:**
1. Micro-averaged F1
2. Macro-averaged F1
3. Weighted F1

**Rhythm game specific:**
1. Tier 1 classes (kick, snare, rhythm hand) recall
2. Variation classification accuracy (given correct instrument)
3. False positive rate (frustration metric)

### Qualitative Evaluation

1. **Listen tests:**
   - Sample predictions on test songs
   - Check musical coherence
   - Identify systematic errors

2. **Visualization:**
   - Plot predictions vs ground truth
   - Confusion matrices
   - Error analysis

3. **Edge cases:**
   - Fast fills
   - Ghost notes
   - Simultaneous hits
   - Soft dynamics

---

## Success Criteria

### Minimum Viable (Ship to Rhythm Game)

**Tier 1 (Critical):**
- Kick: >75% recall, >70% precision
- Snare: >75% recall, >70% precision
- Tom: >70% recall, >65% precision
- Cymbal primary: >75% recall, >70% precision

**Tier 2 (Nice to have):**
- Cymbal variations: >60% precision

### Target Performance

**Tier 1:**
- Kick: >85% recall, >75% precision
- Snare: >85% recall, >75% precision
- Tom: >80% recall, >70% precision
- Cymbal primary: >80% recall, >75% precision

**Tier 2:**
- Cymbal variations: >70% precision

### Stretch Goals

**Tier 1:**
- All >90% recall, >80% precision

**Tier 2:**
- Variations >80% precision
- Hihat pedal >50% recall (currently 5.8%)

---

## Risk Mitigation

### Risk 1: Branches Don't Learn Independently

**Symptom:** One branch dominates, others don't improve

**Mitigation:**
- Monitor per-branch losses separately
- Adjust branch weights dynamically
- Consider gradient clipping per branch
- Use separate optimizers if needed

### Risk 2: Conditional Losses Unstable

**Symptom:** Variation losses explode or vanish

**Mitigation:**
- Ensure mask is computed correctly
- Check for zero-sample batches
- Add epsilon to avoid division by zero
- Start with higher weight, then reduce

### Risk 3: Overfitting

**Symptom:** Val loss increases while train loss decreases

**Mitigation:**
- Increase dropout
- Add more augmentation
- Reduce model capacity
- Early stopping

### Risk 4: Class Imbalance

**Symptom:** Model predicts majority class only

**Mitigation:**
- Use weighted loss (already planned)
- Focal loss for hard examples
- Oversample minority classes
- Check data distribution

### Risk 5: Shared Encoder Too Generic

**Symptom:** Branches don't benefit from sharing

**Mitigation:**
- Add frequency-aware attention in encoder
- Allow branches to have small private encoders
- Experiment with shared vs separate

---

## Timeline Summary

**Week 1: Implementation**
- Days 1-2: Data pipeline & label conversion
- Days 3-4: Shared encoder
- Days 5-7: All 5 branches

**Week 2: Integration & Initial Training**
- Days 1-2: Complete model assembly
- Days 3-4: Training pipeline
- Days 5-7: Debug training on subset

**Week 3: Full Training**
- Full dataset training
- Monitoring & adjustment

**Week 4: Evaluation**
- Comprehensive evaluation
- Comparison with baseline
- Report generation

**Total: 4 weeks from start to evaluation**

---

## Next Steps

### Immediate Actions

1. **Review this plan**
   - Confirm architecture is correct (especially tom branch!)
   - Adjust weights/hyperparameters if needed
   - Add any missing requirements

2. **Set up development environment**
   - Create hierarchical model branch in git
   - Set up directory structure
   - Prepare checkpoints directory

3. **Begin Phase 1: Data Pipeline**
   - Implement label conversion
   - Test on small batch
   - Verify correctness

### Questions Before Implementation

1. **Tom classification granularity:**
   - Do you have separate high/mid tom labels in your data?
   - Or should we treat them as one class?

2. **Hihat pedal:**
   - Keep trying to detect it? (currently 5.8% recall)
   - Or disable and focus on other branches?

3. **Training resources:**
   - GPU availability?
   - Time constraints?

4. **Evaluation priority:**
   - Focus on recall (catch all hits)?
   - Or balance with precision (avoid false positives)?

---

**Document Status:** ✅ READY FOR REVIEW
**Next Action:** User review → Begin Phase 1 implementation
**Last Updated:** 2026-02-08
