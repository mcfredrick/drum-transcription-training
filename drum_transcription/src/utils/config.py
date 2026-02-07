"""Configuration utilities for loading and managing config files."""

import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration object with dot notation access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute with dot notation."""
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self.__dict__})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str = "configs/default_config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Config object with dot notation access
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def save_config(config: Config, output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    # Test config loading
    config = load_config()
    print("Loaded config:")
    print(f"Sample rate: {config.audio.sample_rate}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Number of classes: {config.model.n_classes}")
