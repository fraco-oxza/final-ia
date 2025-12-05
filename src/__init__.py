"""
Ordinal Regression Neural Network package.

A neural network built from scratch using NumPy for ordinal classification tasks.
"""

from .config import (
    NetworkConfig,
    OptimizerConfig,
    RegularizationConfig,
    TrainingConfig,
    DataConfig,
    NumericalConfig,
    Config,
    DEFAULT_CONFIG
)
from .data_utils import (
    load_wine_quality_data,
    normalize_features,
    create_ordinal_encoding,
    ordinal_to_class_indices,
    stratified_split
)
from .model import OrdinalNeuralNetwork
from .training import train, evaluate, TrainingHistory
from .visualization import (
    plot_training_history,
    plot_network_architecture,
    plot_confusion_matrix,
    plot_class_distribution,
    print_training_summary
)


__all__ = [
    # Config
    'NetworkConfig',
    'OptimizerConfig',
    'RegularizationConfig',
    'TrainingConfig',
    'DataConfig',
    'NumericalConfig',
    'Config',
    'DEFAULT_CONFIG',
    # Data
    'load_wine_quality_data',
    'normalize_features',
    'create_ordinal_encoding',
    'ordinal_to_class_indices',
    'stratified_split',
    # Model
    'OrdinalNeuralNetwork',
    # Training
    'train',
    'evaluate',
    'TrainingHistory',
    # Visualization
    'plot_training_history',
    'plot_network_architecture',
    'plot_confusion_matrix',
    'plot_class_distribution',
    'print_training_summary',
]
