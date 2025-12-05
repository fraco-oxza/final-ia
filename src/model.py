"""
Ordinal Regression Neural Network implementation from scratch using NumPy.

Features:
- Leaky ReLU activation for hidden layers
- Sigmoid output layer for cumulative probabilities
- Binary cross-entropy loss for ordinal regression
- AdamW optimizer with decoupled weight decay
- Dropout regularization
- He initialization
"""

from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray

from .config import (
    NetworkConfig,
    OptimizerConfig,
    RegularizationConfig,
    NumericalConfig,
    DEFAULT_CONFIG
)


class OrdinalNeuralNetwork:
    """
    Neural network for ordinal regression using cumulative probability approach.
    
    Instead of predicting P(class = k) for each class, predicts P(class > k)
    for k = 0, 1, ..., K-2, where K is the number of classes.
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        network_config: NetworkConfig = DEFAULT_CONFIG.network,
        optimizer_config: OptimizerConfig = DEFAULT_CONFIG.optimizer,
        regularization_config: RegularizationConfig = DEFAULT_CONFIG.regularization,
        numerical_config: NumericalConfig = DEFAULT_CONFIG.numerical
    ):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            num_classes: Number of output classes K (network outputs K-1 thresholds)
            network_config: Network architecture configuration
            optimizer_config: AdamW optimizer configuration
            regularization_config: Regularization configuration
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        
        # Store configs
        self.network_config = network_config
        self.optimizer_config = optimizer_config
        self.regularization_config = regularization_config
        self.numerical_config = numerical_config
        
        # Build layer sizes: [input, hidden..., output]
        self.layer_sizes = (
            [input_size] + 
            list(network_config.hidden_layers) + 
            [self.num_thresholds]
        )
        self.num_layers = len(self.layer_sizes)
        
        # Training state
        self.training = True
        self.timestep = 0  # For AdamW bias correction
        self.current_learning_rate = optimizer_config.learning_rate
        
        # Initialize weights with He initialization
        self._initialize_weights()
        
        # Cache for forward/backward pass
        self.activations: List[NDArray[np.float64]] = []
        self.pre_activations: List[NDArray[np.float64]] = []
        self.dropout_masks: List[NDArray[np.float64] | None] = []
    
    def _initialize_weights(self) -> None:
        """Initialize weights using He initialization and biases to zero."""
        self.weights: List[NDArray[np.float64]] = []
        self.biases: List[NDArray[np.float64]] = []
        
        # AdamW momentum and velocity
        self.m_weights: List[NDArray[np.float64]] = []
        self.v_weights: List[NDArray[np.float64]] = []
        self.m_biases: List[NDArray[np.float64]] = []
        self.v_biases: List[NDArray[np.float64]] = []
        
        for i in range(self.num_layers - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            
            # He initialization: sqrt(2 / fan_in)
            std = np.sqrt(self.numerical_config.he_init_factor / fan_in)
            w = np.random.randn(fan_in, fan_out) * std
            b = np.zeros((1, fan_out))
            
            self.weights.append(w)
            self.biases.append(b)
            
            # Initialize Adam moments to zero
            self.m_weights.append(np.zeros_like(w))
            self.v_weights.append(np.zeros_like(w))
            self.m_biases.append(np.zeros_like(b))
            self.v_biases.append(np.zeros_like(b))
    
    def _leaky_relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Leaky ReLU activation: f(x) = x if x > 0, else alpha * x."""
        alpha = self.network_config.leaky_relu_alpha
        return np.where(z > 0, z, alpha * z)
    
    def _leaky_relu_derivative(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Derivative of Leaky ReLU."""
        alpha = self.network_config.leaky_relu_alpha
        return np.where(z > 0, 1.0, alpha)
    
    def _sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sigmoid activation for output layer."""
        # Clip for numerical stability
        z_clipped = np.clip(
            z,
            self.numerical_config.sigmoid_clip_min,
            self.numerical_config.sigmoid_clip_max
        )
        return 1.0 / (1.0 + np.exp(-z_clipped))
    
    def forward(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Cumulative probabilities P(Y > k) of shape (n_samples, K-1)
        """
        self.activations = [X]
        self.pre_activations = []
        self.dropout_masks = []
        
        current = X
        dropout_rate = self.regularization_config.dropout_rate
        
        # Hidden layers with Leaky ReLU and Dropout
        for i in range(self.num_layers - 2):
            z = current @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            current = self._leaky_relu(z)
            
            # Apply inverted dropout during training
            if self.training and dropout_rate > 0:
                mask = (np.random.rand(*current.shape) > dropout_rate).astype(np.float64)
                mask /= (1 - dropout_rate)  # Scale to maintain expected value
                current = current * mask
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)
            
            self.activations.append(current)
        
        # Output layer with Sigmoid (cumulative probabilities)
        z = current @ self.weights[-1] + self.biases[-1]
        self.pre_activations.append(z)
        output = self._sigmoid(z)
        self.activations.append(output)
        
        return output
    
    def compute_loss(
        self,
        y_pred: NDArray[np.float64],
        y_true: NDArray[np.float64]
    ) -> float:
        """
        Compute ordinal binary cross-entropy loss.
        
        Args:
            y_pred: Predicted cumulative probabilities (n_samples, K-1)
            y_true: Ordinal encoded labels (n_samples, K-1)
            
        Returns:
            Average loss across all samples and thresholds
        """
        epsilon = self.numerical_config.log_epsilon
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy for each threshold
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) +
            (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return float(loss)
    
    def backward(
        self,
        y_true: NDArray[np.float64]
    ) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]]]:
        """
        Backpropagation to compute gradients.
        
        Args:
            y_true: Ordinal encoded labels (n_samples, K-1)
            
        Returns:
            weight_gradients: List of gradients for each weight matrix
            bias_gradients: List of gradients for each bias vector
        """
        m = y_true.shape[0]  # Batch size
        
        weight_gradients: List[NDArray[np.float64]] = []
        bias_gradients: List[NDArray[np.float64]] = []
        
        # Output layer gradient (sigmoid + BCE): d_loss/d_z = y_pred - y_true
        delta = self.activations[-1] - y_true
        
        # Gradient for output layer weights
        dW = self.activations[-2].T @ delta / m
        db = np.mean(delta, axis=0, keepdims=True)
        weight_gradients.insert(0, dW)
        bias_gradients.insert(0, db)
        
        # Backpropagate through hidden layers
        for i in range(self.num_layers - 3, -1, -1):
            # Propagate error back through weights
            delta = delta @ self.weights[i + 1].T
            
            # Apply Leaky ReLU derivative
            delta = delta * self._leaky_relu_derivative(self.pre_activations[i])
            
            # Apply dropout mask if it was used
            if self.dropout_masks[i] is not None:
                delta = delta * self.dropout_masks[i]
            
            # Compute gradients
            dW = self.activations[i].T @ delta / m
            db = np.mean(delta, axis=0, keepdims=True)
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
        
        return weight_gradients, bias_gradients
    
    def update_parameters(
        self,
        weight_gradients: List[NDArray[np.float64]],
        bias_gradients: List[NDArray[np.float64]]
    ) -> None:
        """
        Update weights and biases using AdamW optimizer.
        
        AdamW decouples weight decay from the gradient update, providing
        better regularization than standard Adam with L2.
        """
        self.timestep += 1
        
        opt = self.optimizer_config
        lr = self.current_learning_rate
        
        # Bias correction factors
        bias_correction1 = 1 - opt.beta1 ** self.timestep
        bias_correction2 = 1 - opt.beta2 ** self.timestep
        
        for i in range(len(self.weights)):
            # Update first moment (momentum)
            self.m_weights[i] = (
                opt.beta1 * self.m_weights[i] +
                (1 - opt.beta1) * weight_gradients[i]
            )
            self.m_biases[i] = (
                opt.beta1 * self.m_biases[i] +
                (1 - opt.beta1) * bias_gradients[i]
            )
            
            # Update second moment (RMSprop-like)
            self.v_weights[i] = (
                opt.beta2 * self.v_weights[i] +
                (1 - opt.beta2) * (weight_gradients[i] ** 2)
            )
            self.v_biases[i] = (
                opt.beta2 * self.v_biases[i] +
                (1 - opt.beta2) * (bias_gradients[i] ** 2)
            )
            
            # Bias-corrected estimates
            m_hat_w = self.m_weights[i] / bias_correction1
            v_hat_w = self.v_weights[i] / bias_correction2
            m_hat_b = self.m_biases[i] / bias_correction1
            v_hat_b = self.v_biases[i] / bias_correction2
            
            # AdamW update with decoupled weight decay
            self.weights[i] -= lr * (
                m_hat_w / (np.sqrt(v_hat_w) + opt.epsilon) +
                opt.weight_decay * self.weights[i]
            )
            
            # Biases typically don't have weight decay
            self.biases[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + opt.epsilon)
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Make class predictions.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predicted class indices of shape (n_samples,)
        """
        was_training = self.training
        self.training = False
        
        cumulative_probs = self.forward(X)
        
        # Predict class = number of thresholds exceeded (P > threshold)
        threshold = self.numerical_config.prediction_threshold
        predictions = np.sum(cumulative_probs > threshold, axis=1).astype(np.int64)
        
        self.training = was_training
        return predictions
    
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Get cumulative probabilities.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Cumulative probabilities P(Y > k) of shape (n_samples, K-1)
        """
        was_training = self.training
        self.training = False
        
        probs = self.forward(X)
        
        self.training = was_training
        return probs
    
    def compute_accuracy(
        self,
        X: NDArray[np.float64],
        y_ordinal: NDArray[np.float64]
    ) -> float:
        """
        Compute classification accuracy.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            y_ordinal: Ordinal encoded labels of shape (n_samples, K-1)
            
        Returns:
            Accuracy score between 0 and 1
        """
        predictions = self.predict(X)
        true_labels = np.sum(y_ordinal, axis=1).astype(np.int64)
        return float(np.mean(predictions == true_labels))
    
    def get_layer_sizes(self) -> List[int]:
        """Return the layer sizes for visualization."""
        return self.layer_sizes.copy()
