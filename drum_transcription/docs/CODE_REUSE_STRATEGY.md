# Code Reuse Strategy for Hierarchical Model

**Purpose:** Document what existing code we can reuse vs what needs to be new for the hierarchical drum transcription model.

**Goal:** Minimize rewriting - leverage ~70% of existing codebase

---

## What We Can REUSE Directly ✅

### 1. CNN Encoder (100% reusable)

**Source:** `src/models/crnn.py` lines 118-144

```python
# REUSE AS-IS - No changes needed
def _build_cnn_encoder(self, filters, kernel_size, pool_size, dropout):
    """Build CNN encoder blocks."""
    # ... existing code works perfectly for hierarchical model
```

**Why it works:**
- Shared encoder is identical in both architectures
- Input: (batch, 1, n_mels, time)
- Output: (batch, channels, freq', time') - same for all branches

**Action:** Copy method directly into hierarchical model

---

### 2. GRU/RNN Encoder (100% reusable)

**Source:** `src/models/crnn.py` lines 88-95

```python
# REUSE AS-IS
self.gru = nn.GRU(
    input_size=self.cnn_output_size,
    hidden_size=hidden_size,
    num_layers=num_gru_layers,
    batch_first=True,
    bidirectional=bidirectional,
    dropout=dropout_gru if num_gru_layers > 1 else 0
)
```

**Why it works:**
- Shared temporal modeling for all branches
- Single RNN feeds into all branches
- Same hyperparameters

**Action:** Copy initialization and forward pass logic

---

### 3. Utility Functions (100% reusable)

#### Length Adjustment

**Source:** `src/models/crnn.py` lines 173-178

```python
# REUSE AS-IS
def _adjust_lengths_for_pooling(self, lengths):
    """Adjust lengths to account for pooling in time dimension."""
    time_reduction = self.pool_size ** self.num_pools
    adjusted_lengths = lengths // time_reduction
    return adjusted_lengths
```

**Action:** Copy directly

#### Label Downsampling

**Source:** `src/models/crnn.py` lines 209-232

```python
# REUSE AS-IS (might need adaptation for hierarchical labels)
def _downsample_labels(self, labels, target_time):
    """Downsample labels to match model output size."""
    # ... existing interpolation logic
```

**Action:**
- Copy method
- May need separate version for hierarchical label dict

---

### 4. AUC Metrics Computation (95% reusable)

**Source:** `src/models/crnn.py` lines 452-502

```python
# REUSE STRUCTURE, adapt for per-branch
def _compute_auc_metrics(self, predictions, labels):
    """Compute AUC metrics (ROC AUC and PR AUC)."""
    metrics = {}

    # Per-class AUC
    for class_idx in range(self.n_classes):
        y_true = labels[:, class_idx]
        y_pred = predictions[:, class_idx]
        # ... ROC-AUC and PR-AUC computation
```

**Adaptation needed:**
- Loop over branches instead of classes
- Handle hierarchical structure (primary + variations)
- Otherwise logic is identical

**Action:**
- Copy method
- Modify to accept branch predictions dict
- Add per-branch AUC computation

---

### 5. Optimizer Configuration (100% reusable)

**Source:** `src/models/crnn.py` lines 504-527

```python
# REUSE AS-IS
def configure_optimizers(self):
    optimizer = torch.optim.Adam(...)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...)
    return {'optimizer': optimizer, 'lr_scheduler': {...}}
```

**Action:** Copy directly

---

### 6. Training/Validation/Test Step Structure (90% reusable)

**Source:** `src/models/crnn.py` lines 180-368

```python
# REUSE STRUCTURE

def training_step(self, batch, batch_idx):
    specs, labels, lengths = batch
    predictions = self(specs)  # ← Different (multi-branch)
    adjusted_lengths = self._adjust_lengths_for_pooling(lengths)
    labels_downsampled = self._downsample_labels(labels, predictions.size(1))  # ← Different (hierarchical)
    loss = self._compute_masked_loss(...)  # ← Different (multi-branch)
    self.log('train_loss', loss, ...)
    return loss
```

**What stays same:**
- Overall structure
- Batch unpacking
- Length adjustment
- Logging pattern

**What changes:**
- Forward pass returns dict, not tensor
- Label downsampling handles hierarchical labels
- Loss computation loops over branches

**Action:**
- Copy structure
- Modify forward pass, labels, and loss sections

---

### 7. Validation Epoch End Pattern (80% reusable)

**Source:** `src/models/crnn.py` lines 269-302

```python
# REUSE PATTERN

def on_validation_epoch_end(self):
    # Collect predictions and labels
    all_preds = []
    all_labels = []
    for output in self.validation_step_outputs:
        # ... extraction logic

    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute AUC
    auc_metrics = self._compute_auc_metrics(all_preds, all_labels)

    # Log
    for name, value in auc_metrics.items():
        self.log(f'val_{name}', value, ...)
```

**What changes:**
- `all_preds` and `all_labels` are dicts (one per branch)
- Loop over branches
- Different metric names (branch names, not class indices)

**Action:**
- Copy pattern
- Adapt for multi-branch structure

---

## What Needs REFACTORING 🔧

### 1. Forward Pass

**Current:**
```python
def forward(self, x):
    x = self.conv_blocks(x)  # ✅ Keep
    x = x.permute(0, 3, 1, 2).reshape(...)  # ✅ Keep
    x, _ = self.gru(x)  # ✅ Keep
    x = self.fc(x)  # ❌ Change - single output layer
    return x
```

**Hierarchical:**
```python
def forward(self, x):
    # Shared encoding (SAME)
    features = self.conv_blocks(x)
    features = features.permute(0, 3, 1, 2).reshape(...)
    features, _ = self.gru(features)

    # Branch predictions (NEW)
    predictions = {
        'kick': self.kick_branch(features),
        'snare': self.snare_branch(features),
        'tom': self.tom_branch(features),
        'cymbal': self.cymbal_branch(features),
        'crash': self.crash_branch(features)
    }
    return predictions
```

**Complexity:** Low - just replace single FC with branch calls

---

### 2. Loss Computation

**Current:** Single criterion, single loss

**Hierarchical:** Multiple branch losses with weights

```python
def compute_loss(self, predictions, labels, lengths):
    total_loss = 0
    loss_components = {}

    # Kick loss (SIMILAR to current masked loss)
    kick_loss = self._compute_branch_loss(
        predictions['kick'], labels['kick'], lengths
    )
    total_loss += self.branch_weights['kick'] * kick_loss
    loss_components['kick'] = kick_loss.item()

    # ... repeat for each branch

    return total_loss, loss_components
```

**Complexity:** Medium - loop over branches, apply weights

---

### 3. Metrics Computation

**Current:** Single set of metrics across all classes

**Hierarchical:** Per-branch metrics

```python
def _compute_metrics(self, predictions, labels, lengths):
    metrics = {}

    # Kick branch metrics
    kick_metrics = self._compute_branch_metrics(
        predictions['kick'], labels['kick'], lengths
    )
    for name, value in kick_metrics.items():
        metrics[f'kick_{name}'] = value

    # ... repeat for each branch

    return metrics
```

**Complexity:** Medium - wrap existing logic with branch loop

---

## What Needs to be NEW 🆕

### 1. Branch Module Classes

**New files needed:**
- `src/models/branches/kick_branch.py`
- `src/models/branches/snare_branch.py`
- `src/models/branches/tom_branch.py`
- `src/models/branches/cymbal_branch.py`
- `src/models/branches/crash_branch.py`

**Complexity:** Low - simple nn.Sequential modules

**Estimated LOC:** ~30-50 lines per branch (200 total)

---

### 2. Hierarchical Label Conversion

**New file:** `src/data/hierarchical_labels.py`

```python
def convert_to_hierarchical(labels_12class):
    """Convert 12-class labels to hierarchical branch format."""
    # Map classes to branches
    # Handle hierarchical structure (primary + variations)
    return hierarchical_dict
```

**Complexity:** Medium - need to understand label structure

**Estimated LOC:** ~150 lines

---

### 3. Conditional Loss Computation

**New in model:**

```python
def _compute_conditional_loss(self, predictions, labels, mask):
    """
    Compute loss only where condition is true.
    E.g., hihat variation loss only where hihat is detected.
    """
    # Mask predictions and labels
    # Compute loss on subset
```

**Complexity:** Medium - handle variable-size batches

**Estimated LOC:** ~50 lines

---

### 4. Multi-Branch Model Class

**New file:** `src/models/hierarchical_crnn.py`

**Estimated LOC:** ~600 lines
- ~400 from existing CRNN (reused)
- ~200 new (branch-specific logic)

---

## Existing Data Pipeline (Can Reuse Most)

### DataModule: 90% Reusable

**Source:** `src/data/data_module.py`

**What works:**
- Dataset loading
- Train/val/test splits
- Batch collation
- Data loading

**What needs change:**
- `__getitem__` should return hierarchical labels (or convert in collate_fn)

**Action:**
- Keep existing DataModule
- Add label conversion in collate function or preprocessing

---

### Preprocessing: 80% Reusable

**Source:** `scripts/preprocess.py`

**What works:**
- Audio loading
- Spectrogram computation
- MIDI parsing

**What needs change:**
- Add crash cymbal (MIDI 49) to mapping
- Output hierarchical label format

**Action:**
- Update MIDI mapping to include crash
- Modify label output format

---

## Summary: Reuse Percentage

| Component | Reuse % | Lines Reused | Lines New | Total |
|-----------|---------|--------------|-----------|-------|
| CNN Encoder | 100% | 30 | 0 | 30 |
| RNN Encoder | 100% | 20 | 0 | 20 |
| Utilities | 100% | 80 | 10 | 90 |
| Training Loop | 90% | 150 | 20 | 170 |
| AUC Metrics | 95% | 50 | 5 | 55 |
| Optimizer | 100% | 25 | 0 | 25 |
| **Subtotal Reused** | | **355** | | |
| Branch Modules | 0% | 0 | 200 | 200 |
| Label Conversion | 0% | 0 | 150 | 150 |
| Loss Computation | 20% | 20 | 80 | 100 |
| **Subtotal New** | | | **430** | |
| **TOTAL** | **45%** | **375** | **465** | **840** |

**Overall Code Reuse: ~45% of hierarchical model code comes from existing CRNN**

**Actual new code needed: ~465 lines**
- ~200 lines: Branch modules (simple)
- ~150 lines: Label conversion (medium)
- ~115 lines: Loss/metrics adaptation (low)

---

## Implementation Strategy

### Phase 1: Copy & Adapt Encoder (Day 1)

1. Create `src/models/hierarchical_crnn.py`
2. Copy shared encoder code from existing CRNN:
   - `_build_cnn_encoder` ✅
   - GRU initialization ✅
   - `_adjust_lengths_for_pooling` ✅
   - `_downsample_labels` ✅
3. Test forward pass through shared encoder

**Reused:** ~150 lines
**New:** ~50 lines (boilerplate)

---

### Phase 2: Implement Branches (Days 2-3)

1. Copy simple FC patterns from existing CRNN
2. Create 5 branch modules (very similar to each other)
3. Wire branches to shared encoder output

**Reused:** ~30 lines (FC pattern)
**New:** ~200 lines (branch variations)

---

### Phase 3: Adapt Training Loop (Day 4)

1. Copy training/validation/test step structure
2. Modify for multi-branch:
   - Forward pass returns dict
   - Loss sums across branches
   - Metrics computed per branch
3. Copy optimizer config directly

**Reused:** ~200 lines (structure)
**New:** ~100 lines (multi-branch logic)

---

### Phase 4: Label Conversion (Day 5)

1. Copy label format from existing preprocessing
2. Write conversion function (12-class → hierarchical)
3. Test on sample batches

**Reused:** ~20 lines (label structure)
**New:** ~150 lines (conversion logic)

---

### Phase 5: AUC Metrics (Day 6)

1. Copy `_compute_auc_metrics` method
2. Adapt for per-branch computation
3. Copy validation epoch end pattern

**Reused:** ~80 lines
**New:** ~30 lines (branch looping)

---

## Code Migration Checklist

### From Existing CRNN

- [ ] Copy `_build_cnn_encoder` → No changes needed
- [ ] Copy GRU initialization → No changes needed
- [ ] Copy `_adjust_lengths_for_pooling` → No changes needed
- [ ] Copy `_downsample_labels` → Minor adaptation for dict
- [ ] Copy `configure_optimizers` → No changes needed
- [ ] Copy `_compute_auc_metrics` → Adapt for branches
- [ ] Copy training/validation/test structure → Adapt for multi-branch
- [ ] Copy metrics computation → Wrap in branch loop
- [ ] Copy logging patterns → Change names for branches

### New Components

- [ ] Create branch module files
- [ ] Implement hierarchical label conversion
- [ ] Implement multi-branch loss computation
- [ ] Implement conditional loss (variations)
- [ ] Add branch weight configuration
- [ ] Test full forward/backward pass

---

## Testing Strategy

### Unit Tests (Reuse Existing Patterns)

1. **Test shared encoder** (copy test from existing)
2. **Test each branch** (similar to existing FC test)
3. **Test forward pass** (copy structure, adapt for dict)
4. **Test loss computation** (new, but simple)
5. **Test label conversion** (new)

### Integration Tests

1. **Test full training step** (copy from existing)
2. **Test validation** (copy from existing)
3. **Test AUC computation** (copy from existing, adapt)

---

## Benefits of This Approach

1. **Proven Components**
   - CNN encoder already works
   - Training loop already stable
   - Metrics already validated

2. **Reduced Risk**
   - ~45% of code is battle-tested
   - Only new parts need debugging
   - Familiar patterns throughout

3. **Faster Development**
   - Don't reinvent the wheel
   - Copy-paste > rewrite
   - ~2-3 days saved

4. **Maintainability**
   - Similar structure to existing code
   - Easy to understand if you know CRNN
   - Consistent patterns

5. **Performance**
   - Existing optimizations carry over
   - Proven efficient implementations
   - No regression risk

---

## Estimated Timeline with Reuse

**Without reuse (from scratch):** ~2 weeks
**With reuse (this strategy):** ~1 week

| Phase | Without Reuse | With Reuse | Time Saved |
|-------|---------------|------------|------------|
| Encoder | 2 days | 0.5 days | 1.5 days |
| Branches | 2 days | 1.5 days | 0.5 days |
| Training Loop | 2 days | 1 day | 1 day |
| Metrics/AUC | 1 day | 0.5 days | 0.5 days |
| Label Conversion | 1 day | 1 day | 0 days |
| Testing | 2 days | 1 day | 1 day |
| **TOTAL** | **10 days** | **5.5 days** | **4.5 days** |

---

**Conclusion:** By reusing existing code, we can implement the hierarchical model in roughly **half the time** with **much lower risk**.

**Next Step:** Start Phase 1 - copy shared encoder components and verify they work as-is.
