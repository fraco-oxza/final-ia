"""
Configuration constants for the Wine Quality Neural Network.

All hyperparameters and magic numbers are centralized here for easy tuning.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkConfig:
    """Neural network architecture configuration."""
    
    # Layer sizes (input will be determined by data, output by num_classes - 1 for ordinal)
    hidden_layers: tuple[int, ...] = (128, 64, 32)
    
    # Leaky ReLU negative slope
    leaky_relu_alpha: float = 0.01


@dataclass(frozen=True)
class OptimizerConfig:
    """AdamW optimizer configuration."""
    
    learning_rate: float = 0.002
    beta1: float = 0.9  # First moment decay (momentum)
    beta2: float = 0.999  # Second moment decay (RMSprop-like)
    epsilon: float = 1e-8  # Numerical stability
    weight_decay: float = 0.001  # L2 regularization (decoupled)


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop configuration."""
    
    max_epochs: int = 5000
    early_stopping_patience: int = 300
    lr_decay_factor: float = 0.95
    lr_decay_patience: int = 50  # Epochs without improvement before decay
    min_learning_rate: float = 1e-6
    min_delta: float = 1e-6  # Minimum change for early stopping
    print_every: int = 100


@dataclass(frozen=True)
class RegularizationConfig:
    """Regularization configuration."""
    
    dropout_rate: float = 0.3


@dataclass(frozen=True)
class DataConfig:
    """Data processing configuration."""
    
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    
    # UCI ML Repository dataset ID for Wine Quality
    uci_dataset_id: int = 186
    
    # Epsilon for numerical stability in normalization
    normalization_epsilon: float = 1e-8


@dataclass(frozen=True)
class NumericalConfig:
    """Numerical stability constants."""
    
    # Epsilon for log to avoid log(0)
    log_epsilon: float = 1e-15
    
    # Sigmoid clipping bounds to prevent overflow
    sigmoid_clip_min: float = -500.0
    sigmoid_clip_max: float = 500.0
    
    # He initialization scaling factor
    he_init_factor: float = 2.0
    
    # Ordinal prediction threshold (P > threshold means class exceeded)
    prediction_threshold: float = 0.5


@dataclass
class Config:
    """Main configuration container."""
    
    network: NetworkConfig = NetworkConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    training: TrainingConfig = TrainingConfig()
    regularization: RegularizationConfig = RegularizationConfig()
    data: DataConfig = DataConfig()
    numerical: NumericalConfig = NumericalConfig()


# Default configuration instance
DEFAULT_CONFIG = Config()
