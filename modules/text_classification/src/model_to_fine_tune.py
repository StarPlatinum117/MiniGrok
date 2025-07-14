import torch
from torch.nn import Module
from transformers import DistilBertForSequenceClassification


def load_model(model_name: str, num_labels: int, device: str = "cpu") -> Module:
    """
    Load the DistilBERT model for sequence classification with frozen encoder parameters.

    This function prepares the model for fine-tuning to classify queries into different types: rag_retrieval,
    llm_generation, and text_classification. The trained model will later be used as part of MiniGrok multi-modal
    capabilities for classification of user queries in order to call the appropriate module.

    Args:
        model_name: The name of the pre-trained DistilBERT model.
        num_labels: The number of labels for classification.
        device: The device to load the model on (default: "cpu").
    """
    device = torch.device(device)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    # Freeze encoder parameters.
    for name, param in model.distilbert.named_parameters():
        param.requires_grad = False
    model.to(device)
    return model
