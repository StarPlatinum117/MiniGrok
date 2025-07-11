import pathlib

import torch
from torch.nn import Module
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer


def load_model(
        *,
        model_name: str,
        model_path: pathlib.Path,
        num_labels: int,
        device: str = "cpu",
) -> tuple[Module, DistilBertTokenizer]:
    """
    Load the DistilBERT model for sequence classification.

    Returns:
        Module: The loaded DistilBERT model and tokenizer.
    """
    # Load the model and the tokenizer.
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Load the model weights.
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    # Set the model for eval.
    model.eval()

    return model, tokenizer
