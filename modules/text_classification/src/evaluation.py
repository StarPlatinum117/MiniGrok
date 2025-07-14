import logging
from typing import Callable

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_classifier(model: Module,
                        data_loader: DataLoader,
                        loss_fn: Callable,
                        device: str) -> tuple[float, float]:
    """
    Evaluates a text classifier model on a given dataset.

    Parameters:
        model: The model to evaluate.
        data_loader: DataLoader for the dataset to evaluate on.
        loss_fn: Loss function to use during evaluation.
        device: Device to run the evaluation on (CPU or GPU).

    Returns:
        The average loss and accuracy of the model on the dataset.
    """
    device = torch.device(device)
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Run the model.
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)  # avg over batches
    accuracy = total_correct / total_samples  # avg over all samples

    logging.info(
        f"Evaluation on test set completed: \n"
        f"Avg Loss = {avg_loss: .4f}, Accuracy = {accuracy: .4f}"
    )
    return avg_loss, accuracy
