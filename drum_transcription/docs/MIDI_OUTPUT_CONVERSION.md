# MIDI Output Conversion from Hierarchical Model

## Overview

The hierarchical model outputs **frame-level probabilities** for each drum type. These need to be converted to **MIDI note-on/note-off events** for use in rhythm games or music production.

## Model Output Format

```python
# Model forward pass output
predictions = model(spectrogram)  # (batch, time_frames, ...)

# predictions contains:
{
    'kick': (batch, time, 1),                    # Probabilities [0, 1]
    'snare': (batch, time, 3),                   # Class probabilities
    'tom': {
        'primary': (batch, time, 2),             # tom/no_tom
        'variation': (batch, time, 3)            # floor/high/mid
    },
    'cymbal': {
        'primary': (batch, time, 3),             # none/hihat/ride
        'hihat_variation': (batch, time, 2),     # open/closed
        'ride_variation': (batch, time, 2)       # bell/body
    },
    'crash': (batch, time, 1)                    # Probabilities [0, 1]
}
```

## Conversion Pipeline

### Step 1: Onset Detection (Frame → Events)

Convert continuous probabilities to discrete onset events:

```python
def detect_onsets(probabilities, threshold=0.5, min_interval_frames=3):
    """
    Convert frame-level probabilities to onset events.

    Args:
        probabilities: (time,) array of probabilities
        threshold: Minimum probability for onset
        min_interval_frames: Minimum frames between consecutive onsets

    Returns:
        List of onset frame indices
    """
    # Threshold
    binary = probabilities > threshold

    # Find onset boundaries (0 → 1 transitions)
    onsets = []
    in_onset = False
    onset_start = None

    for i, active in enumerate(binary):
        if active and not in_onset:
            # Start of onset
            onset_start = i
            in_onset = True
        elif not active and in_onset:
            # End of onset - record the peak
            onset_region = probabilities[onset_start:i]
            peak_idx = onset_start + np.argmax(onset_region)
            onsets.append(peak_idx)
            in_onset = False

    # Handle onset at end
    if in_onset:
        onset_region = probabilities[onset_start:]
        peak_idx = onset_start + np.argmax(onset_region)
        onsets.append(peak_idx)

    # Enforce minimum interval
    filtered_onsets = []
    last_onset = -min_interval_frames

    for onset in onsets:
        if onset - last_onset >= min_interval_frames:
            filtered_onsets.append(onset)
            last_onset = onset

    return filtered_onsets
```

### Step 2: Frame Index → Time Conversion

```python
def frames_to_seconds(frame_indices, hop_length=512, sr=22050):
    """
    Convert frame indices to timestamps in seconds.

    Args:
        frame_indices: List of frame indices
        hop_length: STFT hop length (from preprocessing)
        sr: Sample rate

    Returns:
        List of timestamps in seconds
    """
    return [idx * hop_length / sr for idx in frame_indices]
```

### Step 3: Hierarchical Classification

For hierarchical branches (tom, cymbal), determine the specific drum type:

```python
def classify_hierarchical_onset(
    primary_probs,
    variation_probs,
    onset_frame,
    primary_classes=['none', 'hihat', 'ride'],
    variation_classes={'hihat': ['closed', 'open'], 'ride': ['body', 'bell']}
):
    """
    Classify a detected onset in hierarchical branch.

    Args:
        primary_probs: (time, n_primary_classes) probabilities
        variation_probs: dict with {type: (time, n_var_classes)} probabilities
        onset_frame: Frame index of onset
        primary_classes: Names of primary classes
        variation_classes: Dict mapping primary class to variation class names

    Returns:
        (primary_class, variation_class) tuple
    """
    # Get primary class
    primary_idx = np.argmax(primary_probs[onset_frame])
    primary_class = primary_classes[primary_idx]

    if primary_class == 'none' or primary_class not in variation_classes:
        return (primary_class, None)

    # Get variation class
    var_probs = variation_probs[primary_class][onset_frame]
    var_idx = np.argmax(var_probs)
    variation_class = variation_classes[primary_class][var_idx]

    return (primary_class, variation_class)
```

### Step 4: MIDI Note Mapping

Map drum types to MIDI note numbers (General MIDI standard):

```python
# General MIDI Drum Map (MIDI note numbers)
MIDI_NOTE_MAP = {
    'kick': 36,              # Bass Drum 1
    'snare_head': 38,        # Acoustic Snare
    'snare_rim': 37,         # Side Stick / Rimshot
    'floor_tom': 41,         # Low Floor Tom
    'high_tom': 48,          # Hi Mid Tom
    'mid_tom': 47,           # Low-Mid Tom
    'hihat_closed': 42,      # Closed Hi-Hat
    'hihat_open': 46,        # Open Hi-Hat
    'ride': 51,              # Ride Cymbal 1
    'ride_bell': 53,         # Ride Bell
    'crash': 49,             # Crash Cymbal 1
}

# Alternative: Roland TD-17 MIDI mapping
# (Use actual MIDI note numbers from your TD-17 data)
ROLAND_MIDI_NOTE_MAP = {
    # ... extract from your training data
}
```

### Step 5: Complete MIDI Conversion

```python
import mido

def predictions_to_midi(predictions, audio_path, output_midi_path,
                       hop_length=512, sr=22050, thresholds=None):
    """
    Convert model predictions to MIDI file.

    Args:
        predictions: Dict of model outputs (probabilities)
        audio_path: Path to original audio (for duration)
        output_midi_path: Where to save MIDI file
        hop_length: STFT hop length
        sr: Sample rate
        thresholds: Dict of per-branch thresholds (or use defaults)
    """
    if thresholds is None:
        # Default thresholds (can be optimized per-class)
        thresholds = {
            'kick': 0.3,
            'snare': 0.35,
            'tom': 0.35,
            'cymbal': 0.35,
            'crash': 0.6,  # Higher threshold for crash (avoid false positives)
        }

    # Create MIDI file
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo (extract from audio or use default)
    tempo = mido.bpm2tempo(120)  # 120 BPM default
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    # Track all note events (onset_time, note_number, velocity)
    note_events = []

    # 1. Kick
    kick_onsets = detect_onsets(predictions['kick'].squeeze(),
                                threshold=thresholds['kick'])
    kick_times = frames_to_seconds(kick_onsets, hop_length, sr)
    for time in kick_times:
        note_events.append((time, MIDI_NOTE_MAP['kick'], 100))

    # 2. Snare (head/rim classification)
    snare_probs = predictions['snare']  # (time, 3) - none/head/rim
    snare_head_probs = snare_probs[:, 1]  # head
    snare_rim_probs = snare_probs[:, 2]   # rim

    snare_head_onsets = detect_onsets(snare_head_probs, thresholds['snare'])
    snare_head_times = frames_to_seconds(snare_head_onsets, hop_length, sr)
    for time in snare_head_times:
        note_events.append((time, MIDI_NOTE_MAP['snare_head'], 100))

    snare_rim_onsets = detect_onsets(snare_rim_probs, thresholds['snare'])
    snare_rim_times = frames_to_seconds(snare_rim_onsets, hop_length, sr)
    for time in snare_rim_times:
        note_events.append((time, MIDI_NOTE_MAP['snare_rim'], 90))

    # 3. Tom (hierarchical: detect onset, then classify floor/high/mid)
    tom_onsets = detect_onsets(predictions['tom']['primary'][:, 1],  # tom class
                               threshold=thresholds['tom'])

    for onset_frame in tom_onsets:
        # Classify which tom
        tom_type_probs = predictions['tom']['variation'][onset_frame]
        tom_type_idx = np.argmax(tom_type_probs)
        tom_types = ['floor_tom', 'high_tom', 'mid_tom']
        tom_type = tom_types[tom_type_idx]

        time = onset_frame * hop_length / sr
        note_events.append((time, MIDI_NOTE_MAP[tom_type], 95))

    # 4. Cymbal (hierarchical: hihat/ride, then variation)
    cymbal_primary_probs = predictions['cymbal']['primary']  # (time, 3) - none/hihat/ride

    # Hihat onsets
    hihat_probs = cymbal_primary_probs[:, 1]  # hihat class
    hihat_onsets = detect_onsets(hihat_probs, threshold=thresholds['cymbal'])

    for onset_frame in hihat_onsets:
        # Classify open/closed
        hihat_var_probs = predictions['cymbal']['hihat_variation'][onset_frame]
        is_open = hihat_var_probs[1] > hihat_var_probs[0]  # 0=closed, 1=open
        hihat_type = 'hihat_open' if is_open else 'hihat_closed'

        time = onset_frame * hop_length / sr
        note_events.append((time, MIDI_NOTE_MAP[hihat_type], 85))

    # Ride onsets
    ride_probs = cymbal_primary_probs[:, 2]  # ride class
    ride_onsets = detect_onsets(ride_probs, threshold=thresholds['cymbal'])

    for onset_frame in ride_onsets:
        # Classify bell/body
        ride_var_probs = predictions['cymbal']['ride_variation'][onset_frame]
        is_bell = ride_var_probs[1] > ride_var_probs[0]  # 0=body, 1=bell
        ride_type = 'ride_bell' if is_bell else 'ride'

        time = onset_frame * hop_length / sr
        velocity = 110 if is_bell else 90
        note_events.append((time, MIDI_NOTE_MAP[ride_type], velocity))

    # 5. Crash
    crash_onsets = detect_onsets(predictions['crash'].squeeze(),
                                 threshold=thresholds['crash'])
    crash_times = frames_to_seconds(crash_onsets, hop_length, sr)
    for time in crash_times:
        note_events.append((time, MIDI_NOTE_MAP['crash'], 110))

    # Sort all events by time
    note_events.sort(key=lambda x: x[0])

    # Convert to MIDI messages
    current_time = 0
    for time, note, velocity in note_events:
        # Delta time in ticks
        delta_time = int(mido.second2tick(time - current_time, mid.ticks_per_beat, tempo))

        # Note on
        track.append(mido.Message('note_on', note=note, velocity=velocity,
                                 time=delta_time, channel=9))  # Channel 9 = drums

        # Note off (short duration, drums are percussive)
        track.append(mido.Message('note_off', note=note, velocity=0,
                                 time=12, channel=9))  # 12 ticks ~= 10ms

        current_time = time

    # Save MIDI file
    mid.save(output_midi_path)
    print(f"MIDI file saved to: {output_midi_path}")
    print(f"Total note events: {len(note_events)}")
```

## Usage Example

```python
# After training
model = HierarchicalDrumCRNN.load_from_checkpoint('best.ckpt')
model.eval()

# Process audio file
audio_path = 'test_song.wav'
spectrogram = preprocess_audio(audio_path)  # Your preprocessing

# Get predictions
with torch.no_grad():
    predictions = model(spectrogram)

# Convert to MIDI
output_midi = 'test_song_transcription.mid'
predictions_to_midi(
    predictions,
    audio_path,
    output_midi,
    thresholds={
        'kick': 0.25,    # Optimized thresholds from threshold analysis
        'snare': 0.30,
        'tom': 0.35,
        'cymbal': 0.35,
        'crash': 0.65,
    }
)

# Now you have a MIDI file that can be:
# - Imported into DAWs
# - Used in rhythm games
# - Edited manually
# - Analyzed for timing/performance
```

## Advanced: Velocity Estimation

Currently using fixed velocities. Can be improved:

```python
def estimate_velocity(probabilities, onset_frame, window=3):
    """
    Estimate MIDI velocity from probability values around onset.

    Args:
        probabilities: (time,) array
        onset_frame: Frame index of onset
        window: Frames to look around onset

    Returns:
        MIDI velocity (0-127)
    """
    # Get probability peak around onset
    start = max(0, onset_frame - window)
    end = min(len(probabilities), onset_frame + window + 1)
    region = probabilities[start:end]
    peak_prob = np.max(region)

    # Map probability to velocity (0-127)
    # Higher probability = more confident = higher velocity
    velocity = int(np.clip(peak_prob * 127, 40, 127))

    return velocity
```

## Post-Processing for Rhythm Game

For rhythm game use, you might want additional post-processing:

```python
def quantize_to_grid(note_events, bpm, grid='16th'):
    """
    Quantize note timings to musical grid.

    Helps with rhythm game accuracy (notes aligned to beats).
    """
    beat_duration = 60.0 / bpm  # seconds per beat

    if grid == '16th':
        grid_duration = beat_duration / 4
    elif grid == '8th':
        grid_duration = beat_duration / 2
    # etc.

    quantized_events = []
    for time, note, velocity in note_events:
        # Round to nearest grid point
        quantized_time = round(time / grid_duration) * grid_duration
        quantized_events.append((quantized_time, note, velocity))

    return quantized_events
```

## Testing MIDI Output

```python
# Test MIDI playback
import pygame.midi

pygame.midi.init()
player = pygame.midi.Output(0)
player.set_instrument(0, 9)  # Channel 9 = drums

# Load and play MIDI
mid = mido.MidiFile('output.mid')
for msg in mid.play():
    if msg.type == 'note_on':
        player.note_on(msg.note, msg.velocity, 9)
    elif msg.type == 'note_off':
        player.note_off(msg.note, msg.velocity, 9)

player.close()
pygame.midi.quit()
```

---

**Summary:**
Yes, the model will output MIDI! The conversion pipeline:
1. Model → Frame probabilities
2. Onset detection → Discrete events
3. Frame → Time conversion
4. Hierarchical classification → Specific drum types
5. MIDI note mapping → Standard MIDI file

The MIDI output can be used for rhythm games, DAWs, or further analysis.
