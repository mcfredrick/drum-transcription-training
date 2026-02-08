# Improving 26-Class Drum Transcription Performance

## The Challenge

Expanding from 8 to 26 classes (full Roland TD-17 MIDI standard) introduces several problems:
1. **Class imbalance** - Some drums hit rarely (e.g., ride bell, splash cymbal)
2. **Similar timbres** - Hard to distinguish between similar drums (hi-tom vs mid-tom vs low-tom)
3. **More parameters** - Output layer is 3x larger
4. **Data sparsity** - Less training examples per class

## Solution Strategies

### 1. Hierarchical Classification (Recommended)

Instead of predicting all 26 classes directly, use a two-stage approach:

**Stage 1: Coarse classification (8 classes)**
- Kick, Snare, Hi-hat, Toms, Cymbals, Rim shots, Auxiliary, Other

**Stage 2: Fine-grained classification within groups**
- If "Toms" detected → classify as high/mid/low/floor tom
- If "Cymbals" detected → classify as crash/ride/splash/china/bell
- If "Hi-hat" detected → classify as closed/pedal/open

**Implementation:**

```python
class HierarchicalDrumCRNN(L.LightningModule):
    def __init__(self, n_mels=128, n_coarse=8, n_fine_per_coarse=[1,1,3,4,5,3,2,7]):
        super().__init__()
        
        # Shared encoder
        self.encoder = self._build_encoder(n_mels)
        
        # Coarse classifier (stage 1)
        self.coarse_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_coarse),
            nn.Sigmoid()
        )
        
        # Fine classifiers (stage 2) - one per coarse class
        self.fine_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, n_fine),
                nn.Softmax(dim=-1)  # Mutually exclusive within group
            )
            for n_fine in n_fine_per_coarse
        ])
    
    def forward(self, x):
        # Shared encoding
        features = self.encoder(x)  # (batch, time, 256)
        
        # Stage 1: Coarse prediction
        coarse_pred = self.coarse_classifier(features)  # (batch, time, 8)
        
        # Stage 2: Fine prediction (only for detected coarse classes)
        fine_preds = []
        for i, fine_classifier in enumerate(self.fine_classifiers):
            fine_pred = fine_classifier(features)  # (batch, time, n_fine[i])
            # Weight by coarse prediction
            fine_pred = fine_pred * coarse_pred[:, :, i:i+1]
            fine_preds.append(fine_pred)
        
        return coarse_pred, fine_preds
```

**Advantages:**
- Easier to learn coarse distinctions first
- Fine classifiers only activate when needed
- Better handles class imbalance
- Expected improvement: +10-15% F1

---

### 2. Focal Loss for Class Imbalance

Replace BCE loss with Focal Loss to handle rare classes:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Focal loss modulation
        pt = torch.exp(-bce_loss)  # p_t
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# In your model:
self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

**Why it helps:**
- Focuses on hard-to-classify examples
- Down-weights easy negatives (very common with 26 classes)
- Particularly helps rare classes
- Expected improvement: +5-8% F1

---

### 3. Class Weights Based on Frequency

Weight loss by inverse class frequency:

```python
def compute_class_weights(train_labels, n_classes=26):
    """
    Compute inverse frequency weights for each class.
    
    Args:
        train_labels: All training labels (N, time, n_classes)
        n_classes: Number of drum classes
    
    Returns:
        Tensor of class weights
    """
    # Count positive examples per class
    class_counts = train_labels.sum(dim=(0, 1))  # (n_classes,)
    
    # Compute inverse frequency
    total_positives = class_counts.sum()
    class_weights = total_positives / (n_classes * class_counts)
    
    # Clip extreme weights
    class_weights = torch.clamp(class_weights, min=0.5, max=10.0)
    
    return class_weights

# Use in model:
class_weights = compute_class_weights(train_labels)
self.criterion = nn.BCELoss(weight=class_weights)
```

**Expected improvement:** +3-5% F1

---

### 4. Multi-Scale Architecture

Add parallel branches at different temporal resolutions:

```python
class MultiScaleCRNN(L.LightningModule):
    def __init__(self, n_mels=128, n_classes=26):
        super().__init__()
        
        # Three parallel branches with different hop lengths
        self.branch_fast = self._build_branch(n_mels, hop_length=256)   # ~11ms
        self.branch_medium = self._build_branch(n_mels, hop_length=512) # ~23ms
        self.branch_slow = self._build_branch(n_mels, hop_length=1024)  # ~46ms
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Process at multiple scales
        feat_fast = self.branch_fast(x)
        feat_medium = self.branch_medium(x)
        feat_slow = self.branch_slow(x)
        
        # Upsample to same temporal resolution
        feat_fast = F.interpolate(feat_fast, size=feat_medium.shape[1])
        feat_slow = F.interpolate(feat_slow, size=feat_medium.shape[1])
        
        # Concatenate and fuse
        features = torch.cat([feat_fast, feat_medium, feat_slow], dim=-1)
        output = self.fusion(features)
        
        return output
```

**Why it helps:**
- Fast branch captures quick hits (hi-hat)
- Slow branch captures sustained sounds (cymbals)
- Expected improvement: +7-10% F1

---

### 5. Attention Mechanism

Add attention to focus on relevant frequency bands:

```python
class AttentionCRNN(L.LightningModule):
    def __init__(self, n_mels=128, n_classes=26):
        super().__init__()
        
        self.conv_blocks = self._build_cnn(n_mels)
        self.gru = nn.GRU(...)
        
        # Frequency attention
        self.freq_attention = nn.Sequential(
            nn.Linear(cnn_output_size, cnn_output_size // 4),
            nn.ReLU(),
            nn.Linear(cnn_output_size // 4, cnn_output_size),
            nn.Sigmoid()
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        )
        
        self.fc = nn.Linear(256, n_classes)
    
    def forward(self, x):
        # CNN encoding
        x = self.conv_blocks(x)
        x = x.permute(0, 3, 1, 2).reshape(batch_size, -1, cnn_output_size)
        
        # Apply frequency attention
        attention_weights = self.freq_attention(x)
        x = x * attention_weights
        
        # GRU
        x, _ = self.gru(x)
        
        # Apply temporal attention
        x, _ = self.temporal_attention(x, x, x)
        
        # Output
        x = self.fc(x)
        return torch.sigmoid(x)
```

**Expected improvement:** +5-8% F1

---

### 6. Data Augmentation Specific to 26 Classes

Enhance augmentation to create more varied examples:

```python
class EnhancedDrumAugmentation:
    def __init__(self):
        # Existing augmentations...
        
        # Add spectral augmentations
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=35)
    
    def __call__(self, spec, labels):
        # Apply existing augmentations
        spec, labels = super().__call__(spec, labels)
        
        # SpecAugment (helps generalization)
        if random.random() < 0.5:
            spec = self.freq_masking(spec)
        if random.random() < 0.5:
            spec = self.time_masking(spec)
        
        # Mix different drum hits (mixup for drums)
        if random.random() < 0.3:
            spec, labels = self.drum_mixup(spec, labels)
        
        return spec, labels
    
    def drum_mixup(self, spec, labels):
        """Mix two different drum hits to create hybrid examples"""
        # Find two onsets
        onset_frames = torch.where(labels.sum(dim=1) > 0)[0]
        if len(onset_frames) >= 2:
            idx1, idx2 = random.sample(list(onset_frames), 2)
            alpha = random.uniform(0.3, 0.7)
            
            # Mix spectrograms around onsets
            window = 10  # frames
            spec[:, :, idx1-window:idx1+window] = (
                alpha * spec[:, :, idx1-window:idx1+window] + 
                (1-alpha) * spec[:, :, idx2-window:idx2+window]
            )
        
        return spec, labels
```

**Expected improvement:** +3-5% F1

---

### 7. Ensemble of Specialist Models

Train multiple models, each specialized in different drum groups:

```python
class DrumEnsemble:
    def __init__(self):
        # Model 1: Specialist for toms (4 classes)
        self.tom_model = DrumTranscriptionCRNN(n_classes=4)
        
        # Model 2: Specialist for cymbals (7 classes)
        self.cymbal_model = DrumTranscriptionCRNN(n_classes=7)
        
        # Model 3: Specialist for hi-hat states (3 classes)
        self.hihat_model = DrumTranscriptionCRNN(n_classes=3)
        
        # Model 4: Generalist for other drums
        self.general_model = DrumTranscriptionCRNN(n_classes=12)
    
    def predict(self, audio):
        # Run all specialist models
        tom_preds = self.tom_model(audio)
        cymbal_preds = self.cymbal_model(audio)
        hihat_preds = self.hihat_model(audio)
        general_preds = self.general_model(audio)
        
        # Combine predictions
        combined = self.combine_predictions(
            tom_preds, cymbal_preds, hihat_preds, general_preds
        )
        
        return combined
```

**Advantages:**
- Each model focuses on fewer, similar classes
- Better learning for confusable drums
- Can train different models on different data subsets
- Expected improvement: +8-12% F1

---

### 8. Conditional Random Fields (CRF) Post-processing

Add temporal consistency constraints:

```python
from torchcrf import CRF

class CRNN_CRF(L.LightningModule):
    def __init__(self, n_mels=128, n_classes=26):
        super().__init__()
        
        # CRNN encoder
        self.crnn = DrumTranscriptionCRNN(n_mels, n_classes)
        
        # CRF layer for temporal consistency
        self.crf = CRF(n_classes, batch_first=True)
    
    def forward(self, x, labels=None):
        # Get CRNN predictions (emissions)
        emissions = self.crnn(x)  # (batch, time, n_classes)
        
        if self.training:
            # During training: compute CRF loss
            loss = -self.crf(emissions, labels, reduction='mean')
            return loss
        else:
            # During inference: Viterbi decoding
            predictions = self.crf.decode(emissions)
            return predictions
```

**Why it helps:**
- Enforces temporal consistency (e.g., can't have open hi-hat immediately after closed)
- Learns drum pattern transitions
- Expected improvement: +4-6% F1

---

### 9. Larger Model Capacity

Your current model may be too small for 26 classes:

```python
class LargerDrumCRNN(L.LightningModule):
    def __init__(self, n_mels=128, n_classes=26):
        super().__init__()
        
        # Deeper CNN with more filters
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),   # 32 → 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64 → 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 128 → 256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 4 (NEW)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        
        # Larger GRU
        self.gru = nn.GRU(
            input_size=cnn_output_size,
            hidden_size=256,  # 128 → 256
            num_layers=4,     # 3 → 4
            bidirectional=True,
            dropout=0.3
        )
        
        # Wider output head
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  # 256 → 512 input
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),  # Additional layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
            nn.Sigmoid()
        )
```

**Trade-offs:**
- More parameters: ~5-10M instead of ~1-2M
- Longer training time: ~4-6 days instead of 2-3 days
- Better capacity for 26 classes
- Expected improvement: +5-10% F1

---

### 10. Pretrain on 8 Classes, Fine-tune on 26

Transfer learning approach:

```python
# Stage 1: Train on simplified 8-class problem
model_8class = DrumTranscriptionCRNN(n_classes=8)
train(model_8class, epochs=100)  # Train to convergence

# Stage 2: Expand to 26 classes
model_26class = DrumTranscriptionCRNN(n_classes=26)

# Copy encoder weights
model_26class.conv_blocks.load_state_dict(model_8class.conv_blocks.state_dict())
model_26class.gru.load_state_dict(model_8class.gru.state_dict())

# Freeze encoder initially
for param in model_26class.conv_blocks.parameters():
    param.requires_grad = False
for param in model_26class.gru.parameters():
    param.requires_grad = False

# Train only output head
train(model_26class, epochs=20, lr=1e-3)

# Unfreeze and fine-tune all
for param in model_26class.parameters():
    param.requires_grad = True
train(model_26class, epochs=80, lr=1e-4)
```

**Expected improvement:** +8-12% F1

---

## Recommended Combined Strategy

For best results, combine multiple techniques:

### **Approach 1: Hierarchical + Focal Loss + Larger Model**
```python
# 1. Use hierarchical architecture
# 2. Apply focal loss for class imbalance
# 3. Increase model capacity (4 CNN blocks, 256 hidden GRU)
# Expected improvement: +20-25% F1
```

### **Approach 2: Ensemble of Specialists**
```python
# 1. Train 4 specialist models (toms, cymbals, hi-hats, others)
# 2. Use focal loss for each
# 3. Ensemble predictions
# Expected improvement: +18-22% F1
```

### **Approach 3: Pretrain + Multi-Scale + Attention**
```python
# 1. Pretrain on 8 classes
# 2. Fine-tune on 26 classes with multi-scale architecture
# 3. Add attention mechanism
# Expected improvement: +22-28% F1
```

---

## Implementation Priority

**Quick wins (1-2 days work):**
1. ✅ Focal Loss - Easy to implement, immediate improvement
2. ✅ Class Weights - Compute from training data
3. ✅ Enhanced Augmentation - Add SpecAugment

**Medium effort (1 week):**
4. ✅ Larger Model - More filters/layers
5. ✅ Pretrain on 8 classes - Transfer learning
6. ✅ Attention Mechanism - Add to existing model

**Advanced (2-3 weeks):**
7. ✅ Hierarchical Architecture - Major refactor
8. ✅ Multi-Scale Architecture - Complex implementation
9. ✅ Ensemble Approach - Train multiple models

---

## Debugging Poor Performance

Before implementing new techniques, check:

1. **Data quality**: Are all 26 classes well-represented in E-GMD?
2. **Label mapping**: Did you correctly map all Roland TD-17 notes?
3. **Class imbalance**: Run this check:

```python
# Count samples per class
class_counts = train_labels.sum(dim=(0,1))
print("Samples per class:")
for i, count in enumerate(class_counts):
    print(f"  Class {i}: {count:,} ({100*count/class_counts.sum():.2f}%)")

# Identify rare classes (< 1% of total)
rare_classes = torch.where(class_counts < class_counts.sum() * 0.01)[0]
print(f"\nRare classes (< 1%): {rare_classes.tolist()}")
```

4. **Baseline comparison**: What's the F1 for just the 8 main classes? If that's also low, the problem is elsewhere.

---

## Expected Performance Targets

With these improvements:

**Current (26 classes, basic model):** ~40-50% F1
**After focal loss + class weights:** ~50-60% F1  
**After hierarchical architecture:** ~60-70% F1
**After ensemble + all techniques:** ~70-75% F1

**Note:** 26 classes will always perform worse than 8 classes due to increased difficulty. Target is 70-75% F1 (vs 75-85% for 8 classes).

---

## Roland TD-17 MIDI Mapping Reference

For your 26-class implementation, here's the full mapping:

```python
ROLAND_TD17_MAPPING = {
    # Kicks
    36: "kick",
    
    # Snares
    38: "snare_head",
    40: "snare_rim",
    37: "snare_cross_stick",
    
    # Hi-Hat
    42: "hihat_closed",
    44: "hihat_pedal",
    46: "hihat_open",
    
    # Toms
    48: "tom1_head",        # High tom
    50: "tom1_rim",
    45: "tom2_head",        # Mid tom  
    47: "tom2_rim",
    43: "tom3_head",        # Low tom
    58: "tom3_rim",
    41: "floor_tom_head",
    39: "floor_tom_rim",
    
    # Cymbals
    49: "crash1",
    55: "crash1_choke",
    52: "crash2",
    57: "crash2_choke",
    51: "ride",
    53: "ride_bell",
    59: "ride_edge",
    
    # Auxiliary
    54: "aux1",             # Splash/china/etc
    56: "aux2",
    
    # Other
    33: "click",            # Metronome/click
}
```

Total: 26 classes

Good luck! Start with focal loss + class weights as they're quick wins, then move to hierarchical if needed.

## Analysis of Drum Note Distribution in dataset
Key Insights
Dataset Overview:
- 4,211 MIDI files analyzed
- 1,920,922 total drum notes found
- 23 unique drum note classes currently represented
Class Distribution:
The data is heavily dominated by a few core drum elements:
1. Bass Drum 1 (note 36) - 20.16%
2. Acoustic Snare (note 38) - 18.53%
3. Pedal Hi-Hat (note 44) - 12.61%
4. Closed Hi-Hat (note 42) - 11.29%
5. Ride Cymbal 1 (note 51) - 10.26%
These top 5 notes account for 72.85% of all drum hits.
Recommendation for Class Reduction
Option 1: Aggressive Reduction (Recommended)
Remove 7 extremely rare classes that together make up only 0.089% of the data:
- Note 39 (Hand Clap) - 182 occurrences
- Note 45 (Low Tom) - 704 occurrences
- Note 47 (Low-Mid Tom) - 13 occurrences
- Note 49 (Crash Cymbal 1) - 58 occurrences
- Note 56 (Cowbell) - 12 occurrences
- Note 57 (Crash Cymbal 2) - 115 occurrences
- Note 58 (Vibraslap) - 624 occurrences
This reduces your classes from 23 to 16 while losing essentially no information.
Option 2: Moderate Reduction
If you want more aggressive class balancing, merge any notes with <1% representation (13 classes total, representing only 3.09% of data).
The detailed analysis has been saved to /home/matt/Documents/drum-tranxn/drum_transcription/docs/drum_note_analysis.json for further reference. Would you like me to help you implement this class mapping in your CRNN model?