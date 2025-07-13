import json

import torch

from modules.config import TEXT_CLASSIFICATION_MODEL_CONFIG_FILE as CONFIG_PATH
from modules.text_classification.model_loader import load_model

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

model, tokenizer = load_model(
    model_name=config["model_config"]["name"],
    model_path=config["model_config"]["file_path"],
    num_labels=config["model_config"]["num_labels"],
    device=config["device"]
)


def predict(text: str) -> str:
    """
    Predict the class of the given text using the loaded model.

    Args:
        text: The input text to classify.

    Returns:
        str: The predicted class label.
    """
    # Tokenize the input text.
    encodings = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=config["model_config"]["max_length"],
        return_tensors="pt",
    )
    # Run the model.
    with torch.no_grad():
        outputs = model(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"]
        )
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Map the predicted class index to the label.
    label = config["labels"][str(predicted_class)]  # config requires a str label
    return label
