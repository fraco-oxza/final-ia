"""
Data loading and preprocessing utilities for Wine Quality classification.
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from ucimlrepo import fetch_ucirepo

from .config import DataConfig, DEFAULT_CONFIG


def load_wine_quality_data(
    config: DataConfig = DEFAULT_CONFIG.data
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Load the Wine Quality dataset from UCI ML Repository.
    
    Args:
        config: Data configuration with UCI dataset ID
        
    Returns:
        X: Features array of shape (n_samples, n_features)
        y: Target array of shape (n_samples,)
    """
    wine_quality = fetch_ucirepo(id=config.uci_dataset_id)
    X = wine_quality.data.features.values.astype(np.float64)
    y = wine_quality.data.targets.values.flatten().astype(np.int64)
    return X, y


def normalize_features(
    X: NDArray[np.float64],
    mean: NDArray[np.float64] | None = None,
    std: NDArray[np.float64] | None = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Z-score normalize features.
    
    Args:
        X: Features array of shape (n_samples, n_features)
        mean: Pre-computed mean (for test data normalization)
        std: Pre-computed std (for test data normalization)
        
    Returns:
        X_normalized: Normalized features
        mean: Feature means
        std: Feature standard deviations
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    
    # Add small epsilon to avoid division by zero
    X_normalized = (X - mean) / (std + DEFAULT_CONFIG.data.normalization_epsilon)
    return X_normalized, mean, std


def create_ordinal_encoding(
    y: NDArray[np.int64],
    num_classes: int
) -> NDArray[np.float64]:
    """
    Convert class labels to ordinal encoding.
    
    For K classes, ordinal encoding has K-1 columns.
    y_ordinal[i, k] = 1 if class[i] > k, else 0
    
    Example for classes 0, 1, 2, 3:
        Class 0: [0, 0, 0]  (not > 0, not > 1, not > 2)
        Class 1: [1, 0, 0]  (> 0, not > 1, not > 2)
        Class 2: [1, 1, 0]  (> 0, > 1, not > 2)
        Class 3: [1, 1, 1]  (> 0, > 1, > 2)
    
    Args:
        y: Class indices of shape (n_samples,) with values 0 to num_classes-1
        num_classes: Total number of classes K
        
    Returns:
        y_ordinal: Ordinal encoded labels of shape (n_samples, K-1)
    """
    n_samples = len(y)
    n_thresholds = num_classes - 1
    
    y_ordinal = np.zeros((n_samples, n_thresholds), dtype=np.float64)
    for i in range(n_samples):
        # Set 1 for all thresholds below the class
        y_ordinal[i, :y[i]] = 1.0
    
    return y_ordinal


def ordinal_to_class_indices(y_ordinal: NDArray[np.float64]) -> NDArray[np.int64]:
    """
    Convert ordinal encoding back to class indices.
    
    Args:
        y_ordinal: Ordinal encoded labels of shape (n_samples, K-1)
        
    Returns:
        class_indices: Array of class indices
    """
    return np.sum(y_ordinal, axis=1).astype(np.int64)


def stratified_split(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """
    Split data into train/validation/test sets with stratification.
    
    Maintains the same class distribution in each split as in the original dataset.
    
    Args:
        X: Features of shape (n_samples, n_features)
        y: Ordinal encoded labels of shape (n_samples, K-1)
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    np.random.seed(random_seed)
    
    # Get class indices from ordinal encoding
    class_indices = ordinal_to_class_indices(y)
    unique_classes = np.unique(class_indices)
    
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    
    # Split each class proportionally
    for cls in unique_classes:
        cls_mask = class_indices == cls
        cls_indices_arr = np.where(cls_mask)[0]
        np.random.shuffle(cls_indices_arr)
        
        n_cls = len(cls_indices_arr)
        n_test = max(1, int(n_cls * test_ratio))
        n_val = max(1, int(n_cls * val_ratio))
        
        test_indices.extend(cls_indices_arr[:n_test])
        val_indices.extend(cls_indices_arr[n_test:n_test + n_val])
        train_indices.extend(cls_indices_arr[n_test + n_val:])
    
    # Convert to arrays and shuffle
    train_idx = np.array(train_indices)
    val_idx = np.array(val_indices)
    test_idx = np.array(test_indices)
    
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx]
    )


def prepare_data(
    config: DataConfig = DEFAULT_CONFIG.data
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
    NDArray[np.int64], int
]:
    """
    Load, preprocess, and split the Wine Quality dataset.
    
    Args:
        config: Data configuration
        
    Returns:
        X_train, X_val, X_test: Normalized feature arrays
        y_train, y_val, y_test: Ordinal encoded label arrays
        class_labels: Original class labels (quality scores)
        num_classes: Number of unique classes
    """
    # Load raw data
    X_raw, y_raw = load_wine_quality_data(config)
    
    # Get class information
    class_labels = np.unique(y_raw)
    num_classes = len(class_labels)
    
    # Map quality scores to 0-indexed classes
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    y_indexed = np.array([label_to_idx[label] for label in y_raw], dtype=np.int64)
    
    # Normalize features
    X_normalized, _, _ = normalize_features(X_raw)
    
    # Create ordinal encoding
    y_ordinal = create_ordinal_encoding(y_indexed, num_classes)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X_normalized, y_ordinal,
        val_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_labels, num_classes
