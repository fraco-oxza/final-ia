"""
Neural Network implementation from scratch using NumPy.
- Leaky ReLU activation functions
- Softmax output layer
- Cross-entropy loss
- Backpropagation with AdamW optimizer
- Dropout regularization
- Early stopping
"""

import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from data import X, y


class NeuralNetwork:
    def __init__(self, layer_sizes, alpha=0.01, learning_rate=0.001, 
                 beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01,
                 dropout_rate=0.3):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            alpha: Leaky ReLU parameter (slope for negative values)
            learning_rate: Learning rate for AdamW optimizer
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay coefficient (L2 regularization)
            dropout_rate: Probability of dropping a neuron (0 = no dropout)
        """
        self.layer_sizes = layer_sizes
        self.alpha = alpha  # Leaky ReLU parameter
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.num_layers = len(layer_sizes)
        self.t = 0  # Timestep for bias correction
        self.training = True  # Flag for training vs inference mode
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Initialize AdamW momentum and velocity for weights and biases
        self.m_weights = []  # First moment (momentum)
        self.v_weights = []  # Second moment (velocity)
        self.m_biases = []
        self.v_biases = []
        
        for i in range(self.num_layers - 1):
            # He initialization (good for ReLU-like activations)
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
            
            # Initialize moments to zeros
            self.m_weights.append(np.zeros_like(w))
            self.v_weights.append(np.zeros_like(w))
            self.m_biases.append(np.zeros_like(b))
            self.v_biases.append(np.zeros_like(b))
    
    def leaky_relu(self, z):
        """Leaky ReLU activation function."""
        return np.where(z > 0, z, self.alpha * z)
    
    def leaky_relu_derivative(self, z):
        """Derivative of Leaky ReLU."""
        return np.where(z > 0, 1, self.alpha)
    
    def softmax(self, z):
        """Softmax activation for output layer."""
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Output probabilities
        """
        self.activations = [X]  # Store activations for backprop
        self.z_values = []      # Store pre-activation values
        self.dropout_masks = [] # Store dropout masks for backprop
        
        current = X
        
        # Hidden layers with Leaky ReLU and Dropout
        for i in range(self.num_layers - 2):
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            current = self.leaky_relu(z)
            
            # Apply dropout during training only
            if self.training and self.dropout_rate > 0:
                mask = (np.random.rand(*current.shape) > self.dropout_rate).astype(float)
                mask /= (1 - self.dropout_rate)  # Inverted dropout scaling
                current = current * mask
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)
            
            self.activations.append(current)
        
        # Output layer with Softmax (no dropout)
        z = current @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        output = self.softmax(z)
        self.activations.append(output)
        
        return output
    
    def cross_entropy_loss(self, y_pred, y_true):
        """
        Compute cross-entropy loss.
        
        Args:
            y_pred: Predicted probabilities (n_samples, n_classes)
            y_true: One-hot encoded true labels (n_samples, n_classes)
            
        Returns:
            Average cross-entropy loss
        """
        # Clip to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def backward(self, y_true):
        """
        Backpropagation to compute gradients.
        
        Args:
            y_true: One-hot encoded true labels
            
        Returns:
            Gradients for weights and biases
        """
        m = y_true.shape[0]  # Number of samples
        
        weight_gradients = []
        bias_gradients = []
        
        # Output layer gradient (softmax + cross-entropy combined)
        # The gradient simplifies to (y_pred - y_true)
        delta = self.activations[-1] - y_true
        
        # Gradient for last layer
        dW = self.activations[-2].T @ delta / m
        db = np.mean(delta, axis=0, keepdims=True)
        
        weight_gradients.insert(0, dW)
        bias_gradients.insert(0, db)
        
        # Backpropagate through hidden layers
        for i in range(self.num_layers - 3, -1, -1):
            # Propagate error back
            delta = (delta @ self.weights[i + 1].T) * self.leaky_relu_derivative(self.z_values[i])
            
            # Apply dropout mask if it was used during forward pass
            if self.dropout_masks[i] is not None:
                delta = delta * self.dropout_masks[i]
            
            # Compute gradients
            dW = self.activations[i].T @ delta / m
            db = np.mean(delta, axis=0, keepdims=True)
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
        
        return weight_gradients, bias_gradients
        delta = self.activations[-1] - y_true
        
        # Gradient for last layer
        dW = self.activations[-2].T @ delta / m
        db = np.mean(delta, axis=0, keepdims=True)
        
        weight_gradients.insert(0, dW)
        bias_gradients.insert(0, db)
        
        # Backpropagate through hidden layers
        for i in range(self.num_layers - 3, -1, -1):
            # Propagate error back
            delta = (delta @ self.weights[i + 1].T) * self.leaky_relu_derivative(self.z_values[i])
            
            # Compute gradients
            dW = self.activations[i].T @ delta / m
            db = np.mean(delta, axis=0, keepdims=True)
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """
        Update weights and biases using AdamW optimizer.
        
        AdamW decouples weight decay from the gradient-based update,
        which provides better regularization than L2 regularization in Adam.
        
        Args:
            weight_gradients: List of weight gradients
            bias_gradients: List of bias gradients
        """
        self.t += 1  # Increment timestep
        
        # Bias correction factors
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        for i in range(len(self.weights)):
            # Update weights with AdamW
            # Update biased first moment estimate
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_gradients[i]
            # Update biased second raw moment estimate
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (weight_gradients[i] ** 2)
            
            # Compute bias-corrected estimates
            m_hat_w = self.m_weights[i] / bias_correction1
            v_hat_w = self.v_weights[i] / bias_correction2
            
            # AdamW update: decoupled weight decay
            self.weights[i] -= self.learning_rate * (m_hat_w / (np.sqrt(v_hat_w) + self.epsilon) + 
                                                      self.weight_decay * self.weights[i])
            
            # Update biases with AdamW (typically no weight decay on biases)
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * bias_gradients[i]
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (bias_gradients[i] ** 2)
            
            m_hat_b = self.m_biases[i] / bias_correction1
            v_hat_b = self.v_biases[i] / bias_correction2
            
            self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
    
    def train(self, X, y, X_val=None, y_val=None, epochs=2000, verbose=True,
              early_stopping_patience=200, lr_decay=0.95, lr_decay_every=500):
        """
        Train the neural network with early stopping and learning rate decay.
        
        Args:
            X: Training data
            y: One-hot encoded labels
            X_val: Validation data (optional, for early stopping)
            y_val: Validation labels (optional)
            epochs: Maximum number of training iterations
            verbose: Print loss during training
            early_stopping_patience: Stop if val loss doesn't improve for this many epochs
            lr_decay: Factor to multiply learning rate
            lr_decay_every: Apply lr decay every N epochs
            
        Returns:
            Dictionary with training history
        """
        self.training = True
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        best_weights = None
        best_biases = None
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Learning rate decay
            if lr_decay_every > 0 and (epoch + 1) % lr_decay_every == 0:
                self.learning_rate *= lr_decay
            
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            train_loss = self.cross_entropy_loss(y_pred, y)
            history['train_loss'].append(train_loss)
            
            # Backward pass
            weight_grads, bias_grads = self.backward(y)
            
            # Update parameters
            self.update_parameters(weight_grads, bias_grads)
            
            # Compute training accuracy
            self.training = False  # Disable dropout for evaluation
            train_acc = self.accuracy(X, y)
            history['train_acc'].append(train_acc)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.cross_entropy_loss(val_pred, y_val)
                val_acc = self.accuracy(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    # Save best weights
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\n⚠️  Early stopping at epoch {epoch + 1}! Best epoch: {best_epoch + 1}")
                    # Restore best weights
                    self.weights = best_weights
                    self.biases = best_biases
                    break
            
            self.training = True  # Re-enable dropout for next iteration
            
            if verbose and (epoch + 1) % 100 == 0:
                msg = f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                msg += f", LR: {self.learning_rate:.6f}"
                print(msg)
        
        self.training = False  # Set to inference mode after training
        
        # Restore best weights if we used validation
        if X_val is not None and best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases
            if verbose:
                print(f"✓ Restored best weights from epoch {best_epoch + 1}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predicted class indices
        """
        was_training = self.training
        self.training = False  # Disable dropout for inference
        probabilities = self.forward(X)
        self.training = was_training
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Input data
            
        Returns:
            Predicted probabilities for each class
        """
        was_training = self.training
        self.training = False
        probabilities = self.forward(X)
        self.training = was_training
        return probabilities
    
    def accuracy(self, X, y_onehot):
        """
        Compute classification accuracy.
        
        Args:
            X: Input data
            y_onehot: One-hot encoded true labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        true_labels = np.argmax(y_onehot, axis=1)
        return np.mean(predictions == true_labels)


def preprocess_data(X, y):
    """
    Preprocess the wine quality data.
    
    Args:
        X: Features DataFrame
        y: Target DataFrame
        
    Returns:
        X_normalized, y_onehot, class_labels
    """
    # Convert to numpy arrays
    X_np = X.values.astype(np.float64)
    y_np = y.values.flatten()
    
    # Normalize features (z-score normalization)
    mean = np.mean(X_np, axis=0)
    std = np.std(X_np, axis=0)
    X_normalized = (X_np - mean) / (std + 1e-8)
    
    # Get unique classes and create mapping
    unique_classes = np.unique(y_np)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    
    # One-hot encode labels
    y_indices = np.array([class_to_idx[c] for c in y_np])
    num_classes = len(unique_classes)
    y_onehot = np.zeros((len(y_np), num_classes))
    y_onehot[np.arange(len(y_np)), y_indices] = 1
    
    return X_normalized, y_onehot, unique_classes


def stratified_train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split data into training, validation, and test sets with stratification.
    Each split maintains the same class distribution as the original dataset.
    
    Args:
        X: Features
        y: One-hot encoded labels
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    np.random.seed(random_seed)
    
    # Get class indices from one-hot encoded labels
    class_indices = np.argmax(y, axis=1)
    unique_classes = np.unique(class_indices)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Split each class proportionally
    for cls in unique_classes:
        # Get all indices for this class
        cls_indices = np.where(class_indices == cls)[0]
        np.random.shuffle(cls_indices)
        
        n_cls = len(cls_indices)
        n_test = max(1, int(n_cls * test_ratio))  # At least 1 sample
        n_val = max(1, int(n_cls * val_ratio))    # At least 1 sample
        
        test_indices.extend(cls_indices[:n_test])
        val_indices.extend(cls_indices[n_test:n_test + n_val])
        train_indices.extend(cls_indices[n_test + n_val:])
    
    # Shuffle the indices within each set
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    return (X[train_indices], X[val_indices], X[test_indices],
            y[train_indices], y[val_indices], y[test_indices])


def compute_confusion_matrix(y_true, y_pred, num_classes):
    """
    Compute confusion matrix manually.
    
    Args:
        y_true: True class indices
        y_pred: Predicted class indices
        num_classes: Number of classes
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, class_labels, title='Confusion Matrix', normalize=None):
    """Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_labels: List of class labels
        title: Plot title
        normalize: None for raw counts, 'columns' for precision (by prediction), 
                   'rows' for recall (by true label)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize if requested
    if normalize == 'columns':
        # Normalize by column (predictions)
        col_sums = cm.sum(axis=0, keepdims=True)
        cm_display = np.divide(cm.astype('float'), col_sums, where=col_sums!=0)
        cm_display = np.nan_to_num(cm_display)
        title = title + '\n(Normalized by Prediction)'
    elif normalize == 'rows':
        # Normalize by row (true labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm.astype('float'), row_sums, where=row_sums!=0)
        cm_display = np.nan_to_num(cm_display)
        title = title + '\n(Normalized by True Label)'
    else:
        cm_display = cm.astype('float')
    
    is_normalized = normalize is not None
    
    # Create heatmap
    im = ax.imshow(cm_display, interpolation='nearest', cmap=plt.cm.Blues, 
                   vmin=0, vmax=1 if is_normalized else None)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_labels, yticklabels=class_labels,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm_display.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if is_normalized:
                text = f'{cm_display[i, j]:.1%}'
            else:
                text = format(int(cm[i, j]), 'd')
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black",
                    fontsize=9)
    
    fig.tight_layout()
    return fig


def plot_training_loss(losses, title='Training Loss Over Epochs'):
    """Plot training loss curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(losses, 'b-', linewidth=1, alpha=0.7, label='Loss')
    
    # Add smoothed line
    window = min(100, len(losses) // 10)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(losses)), smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_training_history(history, title='Training History'):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=1, alpha=0.7, label='Train Loss')
    if history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=1, alpha=0.7, label='Val Loss')
        # Mark best epoch
        best_epoch = np.argmin(history['val_loss']) + 1
        best_val_loss = min(history['val_loss'])
        axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
        axes[0].scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Over Training', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', linewidth=1, alpha=0.7, label='Train Accuracy')
    if history['val_acc']:
        axes[1].plot(epochs, history['val_acc'], 'r-', linewidth=1, alpha=0.7, label='Val Accuracy')
        # Mark best epoch (by val loss, not val acc)
        best_epoch = np.argmin(history['val_loss']) + 1
        best_val_acc = history['val_acc'][best_epoch - 1]
        axes[1].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
        axes[1].scatter([best_epoch], [best_val_acc], color='g', s=100, zorder=5)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy Over Training', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    fig.tight_layout()
    return fig


def plot_class_distribution(y_true, y_pred, class_labels, title='Class Distribution'):
    """Plot distribution of true vs predicted classes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # True distribution
    true_counts = np.bincount(y_true, minlength=len(class_labels))
    axes[0].bar(range(len(class_labels)), true_counts, color='steelblue', alpha=0.8)
    axes[0].set_xticks(range(len(class_labels)))
    axes[0].set_xticklabels(class_labels)
    axes[0].set_xlabel('Quality Class', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('True Class Distribution', fontsize=14)
    
    # Add count labels on bars
    for i, v in enumerate(true_counts):
        axes[0].text(i, v + 5, str(v), ha='center', fontsize=9)
    
    # Predicted distribution
    pred_counts = np.bincount(y_pred, minlength=len(class_labels))
    axes[1].bar(range(len(class_labels)), pred_counts, color='darkorange', alpha=0.8)
    axes[1].set_xticks(range(len(class_labels)))
    axes[1].set_xticklabels(class_labels)
    axes[1].set_xlabel('Quality Class', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Predicted Class Distribution', fontsize=14)
    
    for i, v in enumerate(pred_counts):
        axes[1].text(i, v + 5, str(v), ha='center', fontsize=9)
    
    fig.tight_layout()
    return fig


def plot_per_class_metrics(cm, class_labels):
    """Plot precision, recall, and F1-score per class."""
    num_classes = len(class_labels)
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(num_classes)
    width = 0.25
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='darkorange', alpha=0.8)
    bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='forestgreen', alpha=0.8)
    
    ax.set_xlabel('Quality Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)
    
    fig.tight_layout()
    return fig


def plot_prediction_confidence(nn, X_test, y_test, class_labels):
    """Plot confidence distribution for correct vs incorrect predictions."""
    probabilities = nn.forward(X_test)
    predictions = np.argmax(probabilities, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    
    # Get max probability (confidence) for each prediction
    confidences = np.max(probabilities, axis=1)
    
    correct_mask = predictions == true_labels
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of confidence scores
    axes[0].hist(correct_conf, bins=30, alpha=0.7, label=f'Correct ({len(correct_conf)})', color='forestgreen')
    axes[0].hist(incorrect_conf, bins=30, alpha=0.7, label=f'Incorrect ({len(incorrect_conf)})', color='crimson')
    axes[0].set_xlabel('Prediction Confidence', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Confidence Distribution: Correct vs Incorrect', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy by confidence bin
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = (predictions[mask] == true_labels[mask]).mean()
            bin_accuracies.append(bin_acc)
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ax2 = axes[1]
    bars = ax2.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.8, color='steelblue')
    ax2.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Calibration: Confidence vs Accuracy', fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_network_architecture(layer_sizes):
    """Visualize the neural network architecture."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    num_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    
    # Calculate positions
    layer_positions = np.linspace(0.1, 0.9, num_layers)
    
    for layer_idx, (x_pos, n_neurons) in enumerate(zip(layer_positions, layer_sizes)):
        # Limit visualization to max 10 neurons per layer
        display_neurons = min(n_neurons, 10)
        y_positions = np.linspace(0.2, 0.8, display_neurons)
        
        for i, y_pos in enumerate(y_positions):
            circle = plt.Circle((x_pos, y_pos), 0.02, color='steelblue', ec='black', linewidth=1)
            ax.add_patch(circle)
        
        # Add "..." if neurons were truncated
        if n_neurons > 10:
            ax.text(x_pos, 0.1, f'...({n_neurons} neurons)', ha='center', fontsize=9)
        
        # Draw connections to next layer
        if layer_idx < num_layers - 1:
            next_neurons = min(layer_sizes[layer_idx + 1], 10)
            next_y = np.linspace(0.2, 0.8, next_neurons)
            for y1 in y_positions[:min(5, len(y_positions))]:
                for y2 in next_y[:min(5, len(next_y))]:
                    ax.plot([x_pos + 0.02, layer_positions[layer_idx + 1] - 0.02],
                           [y1, y2], 'gray', alpha=0.1, linewidth=0.5)
        
        # Layer labels
        layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(num_layers - 2)] + ['Output']
        ax.text(x_pos, 0.92, f'{layer_names[layer_idx]}\n({layer_sizes[layer_idx]})', 
                ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Neural Network Architecture', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Neural Network for Wine Quality Classification")
    print("(with Dropout, Early Stopping, and LR Decay)")
    print("=" * 60)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_normalized, y_onehot, class_labels = preprocess_data(X, y)
    
    print(f"Number of samples: {X_normalized.shape[0]}")
    print(f"Number of features: {X_normalized.shape[1]}")
    print(f"Number of classes (quality levels): {len(class_labels)}")
    print(f"Quality classes: {class_labels}")
    
    # Split data into train/validation/test with stratification
    # This ensures each split has the same class distribution as the original dataset
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X_normalized, y_onehot, val_ratio=0.15, test_ratio=0.15
    )
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Verify stratification: print class distribution in each set
    print("\nClass distribution verification (stratified split):")
    y_train_classes = np.argmax(y_train, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    y_all_classes = np.argmax(y_onehot, axis=1)
    
    print(f"{'Class':<10} {'Full %':<10} {'Train %':<10} {'Val %':<10} {'Test %':<10}")
    print("-" * 50)
    for i, cls in enumerate(class_labels):
        full_pct = 100 * np.sum(y_all_classes == i) / len(y_all_classes)
        train_pct = 100 * np.sum(y_train_classes == i) / len(y_train_classes)
        val_pct = 100 * np.sum(y_val_classes == i) / len(y_val_classes)
        test_pct = 100 * np.sum(y_test_classes == i) / len(y_test_classes)
        print(f"{cls:<10} {full_pct:<10.2f} {train_pct:<10.2f} {val_pct:<10.2f} {test_pct:<10.2f}")
    
    # Define network architecture (bigger network)
    input_size = X_normalized.shape[1]
    output_size = len(class_labels)  # Cardinality of qualities
    hidden_sizes = [128, 64, 32]  # Three hidden layers
    
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    print(f"\nNetwork architecture: {layer_sizes}")
    
    # Create and train network
    print("\nTraining neural network...")
    print("-" * 70)
    
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        alpha=0.01,           # Leaky ReLU slope
        learning_rate=0.002,  # AdamW learning rate (slightly higher)
        beta1=0.9,            # First moment decay
        beta2=0.999,          # Second moment decay
        epsilon=1e-8,         # Numerical stability
        weight_decay=0.001,   # Reduced weight decay
        dropout_rate=0.3      # 30% dropout
    )
    
    history = nn.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=5000,
        verbose=True,
        early_stopping_patience=300,
        lr_decay=0.95,
        lr_decay_every=500
    )
    
    # Evaluate on test set
    print("-" * 70)
    print("\nFinal Evaluation:")
    train_accuracy = nn.accuracy(X_train, y_train)
    val_accuracy = nn.accuracy(X_val, y_val)
    test_accuracy = nn.accuracy(X_test, y_test)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions for plots
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred = nn.predict(X_test)
    
    # Show some predictions
    print("\nSample predictions (first 10 test samples):")
    for i in range(10):
        pred_class = class_labels[y_pred[i]]
        true_class = class_labels[y_test_labels[i]]
        status = "✓" if y_pred[i] == y_test_labels[i] else "✗"
        print(f"  Sample {i+1}: Predicted={pred_class}, Actual={true_class} {status}")
    
    # ==================== PLOTS ====================
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    
    # Create runs folder with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_folder = os.path.join('runs', timestamp)
    os.makedirs(run_folder, exist_ok=True)
    print(f"📁 Saving to: {run_folder}/")
    
    # 1. Plot Network Architecture
    fig_arch = plot_network_architecture(layer_sizes)
    fig_arch.savefig(os.path.join(run_folder, 'plot_architecture.png'), dpi=150, bbox_inches='tight')
    print("✓ Saved: plot_architecture.png")
    
    # 2. Plot Training Loss (with validation)
    fig_loss = plot_training_history(history)
    fig_loss.savefig(os.path.join(run_folder, 'plot_training_history.png'), dpi=150, bbox_inches='tight')
    print("✓ Saved: plot_training_history.png")
    
    # 3. Plot Confusion Matrices
    cm = compute_confusion_matrix(y_test_labels, y_pred, len(class_labels))
    
    # 3a. Raw counts (to see actual numbers)
    fig_cm_raw = plot_confusion_matrix(cm, class_labels, title='Confusion Matrix (Test Set)', normalize=None)
    fig_cm_raw.savefig(os.path.join(run_folder, 'plot_confusion_matrix_raw.png'), dpi=150, bbox_inches='tight')
    print("✓ Saved: plot_confusion_matrix_raw.png (counts)")
    
    # 3b. Normalized by rows (Recall - what % of each true class was classified as what)
    fig_cm_recall = plot_confusion_matrix(cm, class_labels, title='Confusion Matrix (Test Set)', normalize='rows')
    fig_cm_recall.savefig(os.path.join(run_folder, 'plot_confusion_matrix_recall.png'), dpi=150, bbox_inches='tight')
    print("✓ Saved: plot_confusion_matrix_recall.png (normalized by true label)")
    
    # 3c. Normalized by columns (Precision - what % of predictions for each class were correct)
    fig_cm_precision = plot_confusion_matrix(cm, class_labels, title='Confusion Matrix (Test Set)', normalize='columns')
    fig_cm_precision.savefig(os.path.join(run_folder, 'plot_confusion_matrix_precision.png'), dpi=150, bbox_inches='tight')
    print("✓ Saved: plot_confusion_matrix_precision.png (normalized by prediction)")
    
    # 4. Plot Class Distribution
    fig_dist = plot_class_distribution(y_test_labels, y_pred, class_labels)
    fig_dist.savefig(os.path.join(run_folder, 'plot_class_distribution.png'), dpi=150, bbox_inches='tight')
    print("✓ Saved: plot_class_distribution.png")
    
    # 5. Plot Per-Class Metrics
    fig_metrics = plot_per_class_metrics(cm, class_labels)
    fig_metrics.savefig(os.path.join(run_folder, 'plot_per_class_metrics.png'), dpi=150, bbox_inches='tight')
    print("✓ Saved: plot_per_class_metrics.png")
    
    # 6. Plot Prediction Confidence
    fig_conf = plot_prediction_confidence(nn, X_test, y_test, class_labels)
    fig_conf.savefig(os.path.join(run_folder, 'plot_confidence.png'), dpi=150, bbox_inches='tight')
    print("✓ Saved: plot_confidence.png")
    
    print(f"\n✅ All visualizations saved to {run_folder}/")
    
    # Show all plots
    plt.show()
    
    return nn, history


if __name__ == "__main__":
    nn, losses = main()
