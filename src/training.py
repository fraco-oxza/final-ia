"""
Training utilities for the Ordinal Regression Neural Network.

Provides the main training loop with:
- Early stopping
- Learning rate decay
- Progress monitoring
- History tracking
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .model import OrdinalNeuralNetwork
from .config import TrainingConfig, DEFAULT_CONFIG


@dataclass
class TrainingHistory:
    """Container for training metrics history."""
    train_losses: list[float]
    val_losses: list[float]
    train_accuracies: list[float]
    val_accuracies: list[float]
    learning_rates: list[float]
    best_epoch: int
    epochs_trained: int
    
    @property
    def best_val_loss(self) -> float:
        """Return the best validation loss achieved."""
        return min(self.val_losses)
    
    @property
    def best_val_accuracy(self) -> float:
        """Return the best validation accuracy achieved."""
        return max(self.val_accuracies)


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    
    Monitors validation loss and stops training if no improvement
    is seen for a specified number of epochs (patience).
    """
    
    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.best_epoch = 0
        self.should_stop = False
        
        # Store best weights
        self.best_weights: Optional[list[NDArray[np.float64]]] = None
        self.best_biases: Optional[list[NDArray[np.float64]]] = None
    
    def __call__(
        self,
        val_loss: float,
        model: OrdinalNeuralNetwork,
        epoch: int
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: The neural network model
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            
            # Save best weights
            if self.restore_best_weights:
                self.best_weights = [w.copy() for w in model.weights]
                self.best_biases = [b.copy() for b in model.biases]
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def restore_weights(self, model: OrdinalNeuralNetwork) -> None:
        """Restore the best weights to the model."""
        if self.best_weights is not None and self.best_biases is not None:
            model.weights = [w.copy() for w in self.best_weights]
            model.biases = [b.copy() for b in self.best_biases]


class LearningRateScheduler:
    """
    Learning rate scheduler with decay on plateau.
    
    Reduces learning rate when validation loss stops improving.
    """
    
    def __init__(
        self,
        initial_lr: float,
        decay_factor: float,
        decay_patience: int,
        min_lr: float
    ):
        """
        Initialize the scheduler.
        
        Args:
            initial_lr: Starting learning rate
            decay_factor: Factor to multiply LR by on decay
            decay_patience: Epochs to wait before decaying
            min_lr: Minimum allowed learning rate
        """
        self.current_lr = initial_lr
        self.decay_factor = decay_factor
        self.decay_patience = decay_patience
        self.min_lr = min_lr
        
        self.best_loss: Optional[float] = None
        self.counter = 0
    
    def step(self, val_loss: float) -> float:
        """
        Check if learning rate should be reduced.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            Updated learning rate
        """
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.decay_patience:
                self.current_lr = max(
                    self.current_lr * self.decay_factor,
                    self.min_lr
                )
                self.counter = 0
        
        return self.current_lr


def train(
    model: OrdinalNeuralNetwork,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_val: NDArray[np.float64],
    y_val: NDArray[np.float64],
    training_config: TrainingConfig = DEFAULT_CONFIG.training,
    verbose: bool = True
) -> TrainingHistory:
    """
    Train the neural network with early stopping and learning rate decay.
    
    Args:
        model: The OrdinalNeuralNetwork to train
        X_train: Training features
        y_train: Training labels (ordinal encoded)
        X_val: Validation features
        y_val: Validation labels (ordinal encoded)
        training_config: Training configuration
        verbose: Whether to print progress
        
    Returns:
        TrainingHistory containing all training metrics
    """
    # Initialize history
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []
    learning_rates: list[float] = []
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=training_config.early_stopping_patience,
        min_delta=training_config.min_delta,
        restore_best_weights=True
    )
    
    # Initialize learning rate scheduler
    lr_scheduler = LearningRateScheduler(
        initial_lr=model.optimizer_config.learning_rate,
        decay_factor=training_config.lr_decay_factor,
        decay_patience=training_config.lr_decay_patience,
        min_lr=training_config.min_learning_rate
    )
    
    # Training loop
    model.training = True
    
    for epoch in range(1, training_config.max_epochs + 1):
        # Forward pass
        y_pred = model.forward(X_train)
        train_loss = model.compute_loss(y_pred, y_train)
        
        # Backward pass
        weight_grads, bias_grads = model.backward(y_train)
        
        # Update parameters
        model.update_parameters(weight_grads, bias_grads)
        
        # Compute validation metrics (in eval mode)
        model.training = False
        y_val_pred = model.forward(X_val)
        val_loss = model.compute_loss(y_val_pred, y_val)
        
        train_acc = model.compute_accuracy(X_train, y_train)
        val_acc = model.compute_accuracy(X_val, y_val)
        model.training = True
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        learning_rates.append(model.current_learning_rate)
        
        # Update learning rate
        new_lr = lr_scheduler.step(val_loss)
        model.current_learning_rate = new_lr
        
        # Print progress
        if verbose and epoch % training_config.print_every == 0:
            print(
                f"Epoch {epoch:4d}/{training_config.max_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}% | "
                f"LR: {model.current_learning_rate:.6f}"
            )
        
        # Check early stopping
        if early_stopping(val_loss, model, epoch):
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best epoch: {early_stopping.best_epoch}")
            break
    
    # Restore best weights
    early_stopping.restore_weights(model)
    model.training = False
    
    return TrainingHistory(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        learning_rates=learning_rates,
        best_epoch=early_stopping.best_epoch,
        epochs_trained=len(train_losses)
    )


def evaluate(
    model: OrdinalNeuralNetwork,
    X: NDArray[np.float64],
    y: NDArray[np.float64]
) -> Tuple[float, float]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The trained model
        X: Features
        y: Labels (ordinal encoded)
        
    Returns:
        Tuple of (loss, accuracy)
    """
    was_training = model.training
    model.training = False
    
    y_pred = model.forward(X)
    loss = model.compute_loss(y_pred, y)
    accuracy = model.compute_accuracy(X, y)
    
    model.training = was_training
    return loss, accuracy
