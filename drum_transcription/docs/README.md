# Documentation

This directory contains project documentation and architectural decisions.

## Available Documents

- **[storage-architecture.md](./storage-architecture.md)** - Explains why the project is on SSD while data is on HDD, storage layout, and working with this setup

## Quick Reference

### Project Locations
- **Code (SSD):** `~/Documents/drum-tranxn/drum_transcription/`
- **Data (HDD):** `/mnt/hdd/drum-tranxn/`

### Working Directory
Always work from the SSD location:
```bash
cd ~/Documents/drum-tranxn/drum_transcription
uv run python scripts/train.py --config configs/test_config.yaml
```

All data paths are configured in YAML files to point to HDD automatically.
