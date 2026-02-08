# Roland TD-17 Migration Plan

## Executive Summary

This document outlines the complete migration strategy from the current 8-class E-GMD mapping to the comprehensive 26-class Roland TD-17 mapping. This migration will provide significantly enhanced drum transcription capabilities while maintaining backward compatibility.

## Migration Overview

### Current State
- **8 drum classes** with consolidated MIDI notes
- **E-GMD legacy mapping** with note consolidation
- **Limited expressiveness** (e.g., all hi-hats combined)
- **General MIDI compatibility** but missing Roland-specific features

### Target State
- **26 drum classes** following Roland TD-17 standard
- **Individual MIDI notes** for each drum sound
- **Enhanced expressiveness** (edge/rim distinctions, separate cymbals)
- **Roland TD-17 compatibility** with GM fallback option

## Phase 1: Data Preparation (Week 1)

### 1.1 Update Data Processing Pipeline

**Tasks:**
- ✅ Update `src/data/midi_processing.py` with Roland mapping
- ✅ Create `configs/roland_config.yaml` 
- ✅ Update default config with Roland mapping

**Next Steps:**
- [ ] Modify `scripts/preprocess_egmd.py` to use Roland mapping
- [ ] Create new processed dataset with Roland labels
- [ ] Validate new dataset integrity

**Commands:**
```bash
# Process dataset with Roland mapping
cd /home/matt/Documents/drum-tranxn/drum_transcription
uv run python scripts/preprocess_egmd.py --config configs/roland_config.yaml --create-splits-only

# Process full dataset
uv run python scripts/preprocess_egmd.py --config configs/roland_config.yaml --num-workers 4
```

### 1.2 Dataset Validation

**Validation Steps:**
1. Verify all 26 classes have examples
2. Check class distribution balance
3. Validate MIDI note assignments
4. Test data loading pipeline

**Expected Results:**
- 26 drum classes populated
- Class distribution similar to analysis results
- No data corruption during processing

## Phase 2: Model Architecture Updates (Week 2)

### 2.1 Model Modifications

**Required Changes:**
```python
# Current model output
model.n_classes = 8

# New model output  
model.n_classes = 26
```

**Architecture Enhancements:**
- Increase CNN filters: `[32, 64, 128, 256]` (add 4th layer)
- Increase GRU hidden size: `256` (from 128)
- Add dropout: `0.3` CNN, `0.4` GRU
- Adjust training parameters for larger model

### 2.2 Loss Function Updates

**Multi-label Binary Crossentropy:**
```python
# Current: 8-class multi-label
loss = BCELoss()

# New: 26-class multi-label (same loss, more outputs)
loss = BCELoss()
```

**Class Weighting (Optional):**
```python
# Handle class imbalance if needed
class_weights = calculate_class_weights(dataset)
loss = BCELoss(weight=class_weights)
```

### 2.3 Training Configuration

**Updated Parameters:**
- Batch size: `12` (smaller for larger model)
- Learning rate: `0.0008` (slightly lower)
- Epochs: `150` (more for convergence)
- Gradient accumulation: `2` (effective larger batch)

## Phase 3: Model Training (Weeks 3-4)

### 3.1 Training Pipeline

**Training Commands:**
```bash
# Start Roland model training
cd /home/matt/Documents/drum-tranxn/drum_transcription
uv run python scripts/train.py --config configs/roland_config.yaml

# Monitor training with wandb
# Project: drum-transcription-roland
# Experiment: crnn-roland-egmd-v1
```

### 3.2 Training Monitoring

**Key Metrics:**
- Training/validation loss
- Per-class F1 scores
- Confusion matrix (26x26)
- Learning rate schedule

**Success Criteria:**
- Validation loss < 0.08 (target)
- Per-class F1 > 0.7 for major classes
- No catastrophic forgetting

### 3.3 Hyperparameter Tuning

**Tuning Priorities:**
1. Learning rate schedule
2. Class weighting for rare drums
3. Augmentation intensity
4. Model capacity adjustments

## Phase 4: Model Evaluation (Week 5)

### 4.1 Comprehensive Evaluation

**Evaluation Metrics:**
```python
# Per-class metrics
precision_per_class = [P0, P1, ..., P25]
recall_per_class = [R0, R1, ..., R25]
f1_per_class = [F1_0, F1_1, ..., F1_25]

# Overall metrics
macro_f1 = mean(f1_per_class)
weighted_f1 = weighted_mean(f1_per_class)
```

### 4.2 Comparison with Legacy Model

**Comparative Analysis:**
| Metric | Legacy (8-class) | Roland (26-class) | Improvement |
|--------|------------------|-------------------|-------------|
| Macro F1 | 0.82 | TBD | TBD |
| Weighted F1 | 0.85 | TBD | TBD |
| Drum Types | 8 | 26 | +225% |
| Expressiveness | Limited | High | Significant |

### 4.3 Ablation Studies

**Studies to Conduct:**
1. Roland vs GM mapping performance
2. Impact of edge/rim distinctions
3. Class imbalance handling effectiveness
4. Model capacity requirements

## Phase 5: API Integration (Week 6)

### 5.1 API Updates

**Required Changes:**
- ✅ Update API documentation
- [ ] Modify model loading for 26 classes
- [ ] Add mapping standard parameter
- [ ] Update response formats

**API Enhancements:**
```python
# New parameter
mapping_standard = request.form.get('mapping_standard', 'roland')

# Updated response
response['statistics']['per_drum'] = {
    'kick': count_kick,
    'snare_head': count_snare_head,
    # ... all 26 drums
}
```

### 5.2 Backward Compatibility

**Compatibility Layer:**
```python
def map_roland_to_legacy(roland_predictions):
    """Map 26-class predictions to 8-class legacy format"""
    mapping = {
        'kick': 'kick',
        'snare_head': 'snare',
        'snare_xstick': 'snare',
        'snare_rim': 'snare',
        'hihat_closed': 'hihat',
        'hihat_closed_edge': 'hihat',
        'hihat_open': 'hihat',
        'hihat_open_edge': 'hihat',
        'hihat_pedal': 'hihat',
        # ... etc
    }
    return apply_mapping(roland_predictions, mapping)
```

## Phase 6: Deployment & Testing (Week 7)

### 6.1 Deployment Strategy

**Staged Rollout:**
1. **Internal Testing**: Team validation
2. **Beta Testing**: Selected users
3. **Public Release**: Full deployment

**Deployment Checklist:**
- [ ] Model checkpoint validation
- [ ] API endpoint testing
- [ ] Performance benchmarking
- [ ] Documentation completeness
- [ ] Rollback plan prepared

### 6.2 Performance Validation

**Validation Tests:**
```bash
# Test API with Roland mapping
curl -X POST http://localhost:8000/transcribe \
  -F "file@test_audio.wav" \
  -F "mapping_standard=roland"

# Test GM compatibility
curl -X POST http://localhost:8000/transcribe \
  -F "file=test_audio.wav" \
  -F "mapping_standard=gm"
```

**Performance Targets:**
- Inference time: < 2x current model
- Memory usage: < 2x current model
- Accuracy: > legacy model on comparable classes

## Phase 7: Documentation & Training (Week 8)

### 7.1 Documentation Updates

**Required Documentation:**
- ✅ API Reference (completed)
- [ ] Model Architecture Guide
- [ ] Training Tutorial
- [ ] Migration Guide
- [ ] Troubleshooting Guide

### 7.2 Team Training

**Training Topics:**
1. Roland TD-17 mapping overview
2. New model capabilities
3. API parameter changes
4. Troubleshooting common issues

## Risk Assessment & Mitigation

### High-Risk Items

**1. Training Convergence Issues**
- **Risk**: Larger model fails to converge
- **Mitigation**: Progressive training, learning rate scheduling

**2. Class Imbalance**
- **Risk**: Rare drums (tambourine, cowbell) poor performance
- **Mitigation**: Class weighting, data augmentation

**3. Performance Degradation**
- **Risk**: Slower inference, higher memory usage
- **Mitigation**: Model optimization, quantization

### Medium-Risk Items

**1. Data Pipeline Issues**
- **Risk**: Data corruption during reprocessing
- **Mitigation**: Validation checks, backup procedures

**2. API Compatibility**
- **Risk**: Breaking existing integrations
- **Mitigation**: Backward compatibility layer, versioning

## Success Metrics

### Quantitative Metrics
- **Model Performance**: Macro F1 > 0.8
- **Coverage**: 26/26 drum classes functional
- **Performance**: Inference time < 3s per minute
- **Reliability**: 99%+ uptime

### Qualitative Metrics
- **User Satisfaction**: Enhanced expressiveness feedback
- **Adoption Rate**: 80%+ users migrate to new mapping
- **Documentation Quality**: Complete, clear guides

## Timeline Summary

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1 | Data Preparation | Roland-processed dataset |
| 2 | Model Updates | Updated architecture |
| 3-4 | Training | Trained Roland model |
| 5 | Evaluation | Performance metrics |
| 6 | API Integration | Updated endpoints |
| 7 | Deployment | Production rollout |
| 8 | Documentation | Complete guides |

## Resource Requirements

### Hardware
- **GPU Training**: 2x GPUs (3070, 3090) for 2 weeks
- **Storage**: 500GB for processed dataset
- **Memory**: 32GB+ for larger model training

### Software
- **Dependencies**: Updated PyTorch, Lightning
- **Monitoring**: Weights & Biases
- **Validation**: Custom evaluation scripts

### Personnel
- **ML Engineer**: Model development & training
- **Backend Developer**: API integration
- **QA Engineer**: Testing & validation
- **Technical Writer**: Documentation

## Conclusion

This migration plan provides a structured approach to adopting Roland TD-17 mapping while maintaining system stability and backward compatibility. The 26-class mapping will significantly enhance the drum transcription capabilities and provide users with more expressive and detailed MIDI outputs.

The phased approach minimizes risk while ensuring thorough testing and validation at each stage. Upon completion, users will have access to professional-grade drum transcription that matches industry-standard electronic drum kits.
