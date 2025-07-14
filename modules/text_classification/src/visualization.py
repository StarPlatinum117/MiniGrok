import logging
import pathlib

import matplotlib.pyplot as plt


def plot_training_curves(metrics: dict, save_path: pathlib.Path | None = None) -> None:
    """
    Plots training and validation loss and accuracy curves over epochs.

    Parameters:
        metrics: A dictionary containing training and validation metrics.
        save_path: Optional path to save the plots. If None, plots are shown.
    """
    epochs = range(1, len(metrics["training"]["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics["training"]["loss"], linestyle="dashdot", c="blue", label="Train Loss")
    plt.plot(epochs, metrics["validation"]["loss"], linestyle="dashdot", c="orange", label="Val Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics["training"]["accuracy"], linestyle="dashdot", c="blue", label="Train Accuracy")
    plt.plot(epochs, metrics["validation"]["accuracy"], linestyle="dashdot", c="orange", label="Val Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Training curves saved to {save_path}")
    else:
        plt.show()