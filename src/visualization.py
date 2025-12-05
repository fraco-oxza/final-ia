"""
Visualization utilities for the Ordinal Regression Neural Network.

Provides functions for:
- Training history plots (loss and accuracy curves)
- Network architecture visualization
- Confusion matrix display
- Learning rate schedule visualization
"""

from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    learning_rates: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> Tuple[Figure, List[Axes]]:
    """
    Plot training history including loss, accuracy, and optionally learning rate.

    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        train_accuracies: Training accuracy per epoch
        val_accuracies: Validation accuracy per epoch
        learning_rates: Optional learning rate per epoch
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure

    Returns:
        Figure and list of Axes objects
    """
    num_plots = 3 if learning_rates else 2
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    axes_list = list(axes)

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    axes_list[0].plot(epochs, train_losses, "b-", label="Training", linewidth=2)
    axes_list[0].plot(epochs, val_losses, "r--", label="Validation", linewidth=2)
    axes_list[0].set_xlabel("Epoch", fontsize=12)
    axes_list[0].set_ylabel("Loss (BCE)", fontsize=12)
    axes_list[0].set_title("Training and Validation Loss", fontsize=14)
    axes_list[0].legend(fontsize=10)
    axes_list[0].grid(True, alpha=0.3)

    # Accuracy plot
    train_acc_percent = [a * 100 for a in train_accuracies]
    val_acc_percent = [a * 100 for a in val_accuracies]

    axes_list[1].plot(epochs, train_acc_percent, "b-", label="Training", linewidth=2)
    axes_list[1].plot(epochs, val_acc_percent, "r--", label="Validation", linewidth=2)
    axes_list[1].set_xlabel("Epoch", fontsize=12)
    axes_list[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes_list[1].set_title("Training and Validation Accuracy", fontsize=14)
    axes_list[1].legend(fontsize=10)
    axes_list[1].grid(True, alpha=0.3)

    # Learning rate plot (optional)
    if learning_rates and len(axes_list) > 2:
        axes_list[2].plot(epochs, learning_rates, "g-", linewidth=2)
        axes_list[2].set_xlabel("Epoch", fontsize=12)
        axes_list[2].set_ylabel("Learning Rate", fontsize=12)
        axes_list[2].set_title("Learning Rate Schedule", fontsize=14)
        axes_list[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history saved to {save_path}")

    plt.show()

    return fig, axes_list


def plot_network_architecture(
    layer_sizes: List[int],
    figsize: Tuple[int, int] = (14, 8),
    max_neurons_displayed: int = 10,
    save_path: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """
    Visualize the neural network architecture.

    Args:
        layer_sizes: Number of neurons in each layer
        figsize: Figure size (width, height)
        max_neurons_displayed: Maximum neurons to display per layer
        save_path: Optional path to save the figure

    Returns:
        Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    num_layers = len(layer_sizes)
    layer_spacing = 1.0 / (num_layers + 1)

    # Store neuron positions for drawing connections
    neuron_positions: List[List[Tuple[float, float]]] = []

    layer_colors = ["#3498db", "#2ecc71", "#2ecc71", "#2ecc71", "#e74c3c"]
    layer_labels = [
        "Input\n(Features)",
        "Hidden 1",
        "Hidden 2",
        "Hidden 3",
        "Output\n(Thresholds)",
    ]

    for i, layer_size in enumerate(layer_sizes):
        x = (i + 1) * layer_spacing

        # Limit neurons displayed for large layers
        if layer_size > max_neurons_displayed:
            neurons_to_show = max_neurons_displayed
            show_ellipsis = True
        else:
            neurons_to_show = layer_size
            show_ellipsis = False

        neuron_spacing = 0.8 / (neurons_to_show + 1)
        positions = []

        for j in range(neurons_to_show):
            y = 0.1 + (j + 1) * neuron_spacing
            positions.append((x, y))

            # Draw neuron
            color = layer_colors[min(i, len(layer_colors) - 1)]
            circle = Circle((x, y), 0.015, color=color, ec="white", linewidth=2)
            ax.add_patch(circle)

        if show_ellipsis:
            y_ellipsis = 0.5
            ax.text(x, y_ellipsis, "...", fontsize=16, ha="center", va="center")

        neuron_positions.append(positions)

        # Layer label
        label = layer_labels[min(i, len(layer_labels) - 1)]
        ax.text(x, 0.02, f"{label}\n({layer_size})", fontsize=10, ha="center", va="top")

    # Draw connections (only sample for large networks)
    for i in range(len(neuron_positions) - 1):
        from_positions = neuron_positions[i]
        to_positions = neuron_positions[i + 1]

        # Sample connections to avoid clutter
        max_connections = 50
        total_possible = len(from_positions) * len(to_positions)

        if total_possible > max_connections:
            sample_rate = max_connections / total_possible
        else:
            sample_rate = 1.0

        for from_pos in from_positions:
            for to_pos in to_positions:
                if np.random.random() < sample_rate:
                    arrow = FancyArrowPatch(
                        from_pos,
                        to_pos,
                        arrowstyle="-",
                        color="gray",
                        alpha=0.3,
                        linewidth=0.5,
                    )
                    ax.add_patch(arrow)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Ordinal Neural Network Architecture", fontsize=16, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Architecture diagram saved to {save_path}")

    plt.show()

    return fig, ax


def plot_confusion_matrix(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    class_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot a confusion matrix.

    Args:
        y_true: True class indices
        y_pred: Predicted class indices
        class_labels: Optional labels for each class
        figsize: Figure size (width, height)
        cmap: Colormap name
        save_path: Optional path to save the figure

    Returns:
        Figure and Axes objects
    """
    # Compute confusion matrix
    num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    # Default labels if not provided
    if class_labels is None:
        class_labels = [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_labels,
        yticklabels=class_labels,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    plt.show()

    return fig, ax


def plot_class_distribution(
    y_train: NDArray[np.int64],
    y_val: NDArray[np.int64],
    y_test: NDArray[np.int64],
    class_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot the class distribution across train/val/test splits.

    Args:
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
        class_labels: Optional labels for each class
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure

    Returns:
        Figure and Axes objects
    """
    num_classes = max(y_train.max(), y_val.max(), y_test.max()) + 1

    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(num_classes)]

    train_counts = [np.sum(y_train == i) for i in range(num_classes)]
    val_counts = [np.sum(y_val == i) for i in range(num_classes)]
    test_counts = [np.sum(y_test == i) for i in range(num_classes)]

    x = np.arange(num_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width, train_counts, width, label="Train", color="#3498db")
    bars2 = ax.bar(x, val_counts, width, label="Validation", color="#2ecc71")
    bars3 = ax.bar(x + width, test_counts, width, label="Test", color="#e74c3c")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Class Distribution Across Splits", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Class distribution saved to {save_path}")

    plt.show()

    return fig, ax


def print_training_summary(
    train_accuracy: float,
    val_accuracy: float,
    test_accuracy: float,
    final_train_loss: float,
    final_val_loss: float,
    epochs_trained: int,
    best_epoch: int,
) -> None:
    """
    Print a formatted training summary.

    Args:
        train_accuracy: Final training accuracy
        val_accuracy: Final validation accuracy
        test_accuracy: Test set accuracy
        final_train_loss: Final training loss
        final_val_loss: Final validation loss
        epochs_trained: Total epochs trained
        best_epoch: Epoch with best validation loss
    """
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total epochs trained: {epochs_trained}")
    print(f"Best epoch (early stop): {best_epoch}")
    print("-" * 60)
    print(f"Final Training Loss: {final_train_loss:.6f}")
    print(f"Final Validation Loss: {final_val_loss:.6f}")
    print("-" * 60)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("=" * 60 + "\n")
