"""Hierarchical label conversion for multi-branch drum transcription model."""

import torch
import numpy as np
from typing import Dict


def convert_to_hierarchical(labels_12class: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Convert 11 or 12-class labels to hierarchical format for multi-branch model.

    Input shape: (batch, time, 11) or (batch, time, 12) - Multi-label binary format

    Original classes (from drum_config.yaml):
        0: kick
        1: snare_head
        2: snare_rim
        3: side_stick (mapped to snare_rim for hierarchical)
        4: hihat_pedal (DROPPED - not used in hierarchical model)
        5: hihat_closed
        6: hihat_open
        7: floor_tom
        8: high_mid_tom
        9: ride
        10: ride_bell
        11: crash (optional - may not exist in preprocessed data)

    Output: dict with keys for 5 branches:
        'kick': (batch, time) - binary (0 or 1)
        'snare': (batch, time) - 3 classes (0=none, 1=head, 2=rim/side_stick)
        'tom_primary': (batch, time) - 2 classes (0=no_tom, 1=tom)
        'tom_variation': (batch, time) - 3 classes (0=floor, 1=high, 2=mid)
        'cymbal_primary': (batch, time) - 3 classes (0=none, 1=hihat, 2=ride)
        'hihat_variation': (batch, time) - 2 classes (0=closed, 1=open)
        'ride_variation': (batch, time) - 2 classes (0=body, 1=bell)
        'crash': (batch, time) - binary (0 or 1)

    Args:
        labels_12class: Multi-label binary tensor of shape (batch, time, 11 or 12)

    Returns:
        Dictionary of hierarchical labels for each branch
    """
    batch, time, n_classes = labels_12class.shape
    device = labels_12class.device

    hierarchical = {}

    # 1. KICK BRANCH - Binary onset detection (20-100 Hz)
    hierarchical['kick'] = labels_12class[:, :, 0].float()  # (batch, time)

    # 2. SNARE BRANCH - 3-class: none/head/rim (200-1000 Hz + broadband)
    # Combine snare_rim and side_stick into single "rim" class
    snare = torch.zeros(batch, time, dtype=torch.long, device=device)
    snare[labels_12class[:, :, 1] == 1] = 1  # snare_head
    # Combine rim and side_stick
    rim_or_stick = (labels_12class[:, :, 2] == 1) | (labels_12class[:, :, 3] == 1)
    snare[rim_or_stick] = 2  # snare_rim (includes side_stick)
    hierarchical['snare'] = snare  # (batch, time) with values 0, 1, or 2

    # 3. TOM BRANCH - Hierarchical: onset detection + floor/high/mid (80-400 Hz)
    # Primary: Binary tom detection
    tom_mask = (labels_12class[:, :, 7] == 1) | (labels_12class[:, :, 8] == 1)
    tom_primary = torch.zeros(batch, time, dtype=torch.long, device=device)
    tom_primary[tom_mask] = 1  # tom detected
    hierarchical['tom_primary'] = tom_primary  # (batch, time) with values 0 or 1

    # Variation: Which tom? (floor=0, high=1, mid=2)
    # Note: high_mid_tom (class 8) is treated as "high" for now
    # If you have separate high/mid in your data, adjust this
    tom_variation = torch.zeros(batch, time, dtype=torch.long, device=device)
    tom_variation[labels_12class[:, :, 7] == 1] = 0  # floor_tom
    tom_variation[labels_12class[:, :, 8] == 1] = 1  # high_mid_tom -> "high"
    # Note: We don't have separate "mid" tom in the current 12-class system
    hierarchical['tom_variation'] = tom_variation  # (batch, time) with values 0, 1, or 2

    # 4. CYMBAL BRANCH - Rhythm cymbals only (hihat/ride) (2-8 kHz)
    # Primary: Which rhythm cymbal? (none=0, hihat=1, ride=2)
    cymbal_primary = torch.zeros(batch, time, dtype=torch.long, device=device)
    hihat_mask = (labels_12class[:, :, 5] == 1) | (labels_12class[:, :, 6] == 1)
    ride_mask = (labels_12class[:, :, 9] == 1) | (labels_12class[:, :, 10] == 1)
    cymbal_primary[hihat_mask] = 1  # hihat
    cymbal_primary[ride_mask] = 2   # ride
    hierarchical['cymbal_primary'] = cymbal_primary  # (batch, time) with values 0, 1, or 2

    # Hihat variation: open/closed (closed=0, open=1)
    hihat_variation = torch.zeros(batch, time, dtype=torch.long, device=device)
    # Default is closed (0), set to open (1) where hihat_open is detected
    hihat_variation[labels_12class[:, :, 6] == 1] = 1  # open
    hierarchical['hihat_variation'] = hihat_variation  # (batch, time) with values 0 or 1

    # Ride variation: body/bell (body=0, bell=1)
    ride_variation = torch.zeros(batch, time, dtype=torch.long, device=device)
    # Default is body (0), set to bell (1) where ride_bell is detected
    ride_variation[labels_12class[:, :, 10] == 1] = 1  # bell
    hierarchical['ride_variation'] = ride_variation  # (batch, time) with values 0 or 1

    # 5. CRASH BRANCH - Binary accent detector (4-8 kHz)
    # Backward compatibility: if data only has 11 classes, crash is not available
    if n_classes > 11:
        hierarchical['crash'] = labels_12class[:, :, 11].float()  # (batch, time)
    else:
        # Crash not in preprocessed data (old 11-class system), use zeros
        hierarchical['crash'] = torch.zeros(batch, time, dtype=torch.float32, device=device)

    # Note: Hihat pedal (class 4) is DROPPED - not included in hierarchical labels

    return hierarchical


def hierarchical_to_flat(
    hierarchical_preds: Dict[str, torch.Tensor],
    thresholds: Dict[str, float] = None
) -> torch.Tensor:
    """
    Convert hierarchical predictions back to flat 11-class format (without crash).
    Useful for comparison with baseline or generating traditional MIDI output.

    Args:
        hierarchical_preds: Dictionary of predictions from hierarchical model
            - Each value can be logits or probabilities
        thresholds: Optional dict of thresholds per branch (default: 0.5 for all)

    Returns:
        Flat predictions of shape (batch, time, 11) in original class order
        (excluding crash which is in the hierarchical model but not baseline)
    """
    if thresholds is None:
        thresholds = {
            'kick': 0.5,
            'snare': 0.5,
            'tom': 0.5,
            'cymbal': 0.5,
        }

    # Get batch and time dimensions
    kick_pred = hierarchical_preds['kick']
    batch, time = kick_pred.shape[:2]
    device = kick_pred.device

    # Initialize flat predictions (11 classes, no crash)
    flat_preds = torch.zeros(batch, time, 11, device=device)

    # 1. Kick (class 0)
    kick_prob = torch.sigmoid(kick_pred) if kick_pred.max() > 1 else kick_pred
    flat_preds[:, :, 0] = (kick_prob > thresholds['kick']).float()

    # 2-3. Snare (classes 1-2: head, rim)
    snare_probs = torch.softmax(hierarchical_preds['snare'], dim=-1)
    flat_preds[:, :, 1] = (snare_probs[:, :, 1] > thresholds['snare']).float()  # head
    flat_preds[:, :, 2] = (snare_probs[:, :, 2] > thresholds['snare']).float()  # rim
    # Note: side_stick (class 3) is merged with rim, so we leave it as 0

    # 4. Hihat pedal (class 4) - Not predicted by hierarchical model, always 0
    flat_preds[:, :, 4] = 0

    # 5-6. Hihat (classes 5-6: closed, open)
    cymbal_primary_probs = torch.softmax(hierarchical_preds['cymbal']['primary'], dim=-1)
    hihat_detected = cymbal_primary_probs[:, :, 1] > thresholds['cymbal']  # hihat

    hihat_var_probs = torch.softmax(hierarchical_preds['cymbal']['hihat_variation'], dim=-1)
    is_open = hihat_var_probs[:, :, 1] > 0.5  # open

    flat_preds[:, :, 5] = (hihat_detected & ~is_open).float()  # closed
    flat_preds[:, :, 6] = (hihat_detected & is_open).float()   # open

    # 7-8. Toms (classes 7-8: floor, high_mid)
    tom_primary_probs = torch.softmax(hierarchical_preds['tom']['primary'], dim=-1)
    tom_detected = tom_primary_probs[:, :, 1] > thresholds['tom']

    tom_var_probs = torch.softmax(hierarchical_preds['tom']['variation'], dim=-1)
    is_floor = tom_var_probs[:, :, 0] > 0.5

    flat_preds[:, :, 7] = (tom_detected & is_floor).float()   # floor
    flat_preds[:, :, 8] = (tom_detected & ~is_floor).float()  # high_mid

    # 9-10. Ride (classes 9-10: ride, ride_bell)
    ride_detected = cymbal_primary_probs[:, :, 2] > thresholds['cymbal']  # ride

    ride_var_probs = torch.softmax(hierarchical_preds['cymbal']['ride_variation'], dim=-1)
    is_bell = ride_var_probs[:, :, 1] > 0.5

    flat_preds[:, :, 9] = (ride_detected & ~is_bell).float()  # ride body
    flat_preds[:, :, 10] = (ride_detected & is_bell).float()  # ride bell

    return flat_preds


def get_hierarchical_label_info() -> Dict:
    """
    Get information about hierarchical label structure.
    Useful for debugging and validation.

    Returns:
        Dictionary with branch names, shapes, and class counts
    """
    return {
        'branches': {
            'kick': {
                'type': 'binary',
                'output_shape': '(batch, time)',
                'classes': 1,
                'description': 'Binary kick detection (20-100 Hz)'
            },
            'snare': {
                'type': 'multiclass',
                'output_shape': '(batch, time)',
                'classes': 3,
                'class_names': ['none', 'head', 'rim'],
                'description': '3-class snare detection (200-1000 Hz + broadband)'
            },
            'tom_primary': {
                'type': 'binary_multiclass',
                'output_shape': '(batch, time)',
                'classes': 2,
                'class_names': ['no_tom', 'tom'],
                'description': 'Binary tom onset detection (80-400 Hz)'
            },
            'tom_variation': {
                'type': 'multiclass',
                'output_shape': '(batch, time)',
                'classes': 3,
                'class_names': ['floor', 'high', 'mid'],
                'description': 'Tom type classification (conditional on tom_primary)',
                'conditional': True
            },
            'cymbal_primary': {
                'type': 'multiclass',
                'output_shape': '(batch, time)',
                'classes': 3,
                'class_names': ['none', 'hihat', 'ride'],
                'description': 'Rhythm cymbal detection (2-8 kHz)'
            },
            'hihat_variation': {
                'type': 'binary_multiclass',
                'output_shape': '(batch, time)',
                'classes': 2,
                'class_names': ['closed', 'open'],
                'description': 'Hihat open/closed classification (conditional on hihat detection)',
                'conditional': True
            },
            'ride_variation': {
                'type': 'binary_multiclass',
                'output_shape': '(batch, time)',
                'classes': 2,
                'class_names': ['body', 'bell'],
                'description': 'Ride bell/body classification (conditional on ride detection)',
                'conditional': True
            },
            'crash': {
                'type': 'binary',
                'output_shape': '(batch, time)',
                'classes': 1,
                'description': 'Binary crash detection - accent detector (4-8 kHz)'
            }
        },
        'dropped_classes': ['hihat_pedal'],
        'merged_classes': {
            'snare_rim': ['snare_rim', 'side_stick']
        }
    }


if __name__ == "__main__":
    # Test hierarchical label conversion
    print("Testing hierarchical label conversion...")

    # Create sample 12-class labels
    batch_size = 2
    time_steps = 100
    n_classes = 12

    # Random binary labels for testing
    labels_12class = torch.randint(0, 2, (batch_size, time_steps, n_classes)).float()

    print(f"\nInput labels shape: {labels_12class.shape}")
    print(f"Input labels dtype: {labels_12class.dtype}")

    # Convert to hierarchical
    hierarchical = convert_to_hierarchical(labels_12class)

    print("\nHierarchical labels:")
    for key, value in hierarchical.items():
        print(f"  {key:20s}: shape={value.shape}, dtype={value.dtype}, "
              f"unique_values={torch.unique(value).tolist()}")

    # Show label info
    print("\nHierarchical label structure:")
    info = get_hierarchical_label_info()
    for branch_name, branch_info in info['branches'].items():
        print(f"\n  {branch_name}:")
        print(f"    Type: {branch_info['type']}")
        print(f"    Classes: {branch_info['classes']}")
        print(f"    Description: {branch_info['description']}")
        if 'class_names' in branch_info:
            print(f"    Class names: {branch_info['class_names']}")
        if branch_info.get('conditional', False):
            print(f"    Conditional: Yes")

    print(f"\nDropped classes: {info['dropped_classes']}")
    print(f"Merged classes: {info['merged_classes']}")

    print("\n✓ Hierarchical label conversion test completed")
