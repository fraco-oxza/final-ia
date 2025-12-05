#!/usr/bin/env python3
"""
Main entry point for the Ordinal Regression Neural Network.

Trains and evaluates an ordinal neural network on the Wine Quality dataset.
The network predicts wine quality (3-9) using cumulative probability approach.

Usage:
    python main.py
"""

import numpy as np

from src import (
    # Config
    DEFAULT_CONFIG,
    Config,
    NetworkConfig,
    OptimizerConfig,
    RegularizationConfig,
    TrainingConfig,
    # Data
    load_wine_quality_data,
    normalize_features,
    create_ordinal_encoding,
    ordinal_to_class_indices,
    stratified_split,
    # Model
    OrdinalNeuralNetwork,
    # Training
    train,
    evaluate,
    # Visualization
    plot_training_history,
    plot_network_architecture,
    plot_confusion_matrix,
    print_training_summary,
)


def main() -> None:
    """Main function to train and evaluate the ordinal neural network."""
    
    # Set random seed for reproducibility
    np.random.seed(DEFAULT_CONFIG.data.random_seed)
    
    print("=" * 60)
    print("ORDINAL REGRESSION NEURAL NETWORK")
    print("Wine Quality Classification")
    print("=" * 60)
    
    # ==========================================================================
    # Load and preprocess data
    # ==========================================================================
    print("\n[1/5] Loading Wine Quality dataset...")
    
    X, y = load_wine_quality_data()
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Get unique classes and their mapping
    unique_classes = np.sort(np.unique(y))
    num_classes = len(unique_classes)
    print(f"Classes: {unique_classes} ({num_classes} total)")
    
    # Convert labels to 0-indexed
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_indexed = np.array([class_to_idx[label] for label in y])
    
    # Normalize features
    print("\n[2/5] Preprocessing data...")
    X_normalized, mean, std = normalize_features(X)
    
    # Create ordinal encoding
    y_ordinal = create_ordinal_encoding(y_indexed, num_classes)
    print(f"Ordinal encoding shape: {y_ordinal.shape} (K-1 = {num_classes - 1} thresholds)")
    
    # Split data
    splits = stratified_split(
        X_normalized,
        y_ordinal,
        val_ratio=DEFAULT_CONFIG.data.validation_ratio,
        test_ratio=DEFAULT_CONFIG.data.test_ratio,
        random_seed=DEFAULT_CONFIG.data.random_seed
    )
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Val:   {X_val.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    
    # ==========================================================================
    # Create model
    # ==========================================================================
    print("\n[3/5] Creating neural network...")
    
    input_size = X_train.shape[1]
    
    model = OrdinalNeuralNetwork(
        input_size=input_size,
        num_classes=num_classes,
        network_config=DEFAULT_CONFIG.network,
        optimizer_config=DEFAULT_CONFIG.optimizer,
        regularization_config=DEFAULT_CONFIG.regularization
    )
    
    print(f"Architecture: {model.get_layer_sizes()}")
    print(f"Optimizer: AdamW (lr={DEFAULT_CONFIG.optimizer.learning_rate})")
    print(f"Dropout rate: {DEFAULT_CONFIG.regularization.dropout_rate}")
    
    # ==========================================================================
    # Train model
    # ==========================================================================
    print("\n[4/5] Training model...")
    print("-" * 60)
    
    history = train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        training_config=DEFAULT_CONFIG.training,
        verbose=True
    )
    
    # ==========================================================================
    # Evaluate model
    # ==========================================================================
    print("\n[5/5] Evaluating model...")
    
    train_loss, train_acc = evaluate(model, X_train, y_train)
    val_loss, val_acc = evaluate(model, X_val, y_val)
    test_loss, test_acc = evaluate(model, X_test, y_test)
    
    print_training_summary(
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        test_accuracy=test_acc,
        final_train_loss=train_loss,
        final_val_loss=val_loss,
        epochs_trained=history.epochs_trained,
        best_epoch=history.best_epoch
    )
    
    # ==========================================================================
    # Visualize results
    # ==========================================================================
    print("\nGenerating visualizations...")
    
    # Plot training history
    plot_training_history(
        train_losses=history.train_losses,
        val_losses=history.val_losses,
        train_accuracies=history.train_accuracies,
        val_accuracies=history.val_accuracies,
        learning_rates=history.learning_rates,
        save_path='training_history.png'
    )
    
    # Plot network architecture
    plot_network_architecture(
        layer_sizes=model.get_layer_sizes(),
        save_path='network_architecture.png'
    )
    
    # Plot confusion matrix
    y_pred = model.predict(X_test)
    y_true = ordinal_to_class_indices(y_test)
    
    class_labels = [f'Q{c}' for c in unique_classes]
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_labels=class_labels,
        save_path='confusion_matrix.png'
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
