import json
import logging

import torch

from modules.config import TEXT_CLASSIFICATION_MODEL_WEIGHTS_FILE as MODEL_WEIGHTS_FILE
from modules.config import TEXT_CLASSIFICATION_MODEL_CONFIG_FILE as CONFIG_FILE
from modules.text_classification.model_loader import load_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s'
)

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

model, tokenizer = load_model(
    model_name=config["model_config"]["name"],
    model_path=MODEL_WEIGHTS_FILE,
    num_labels=config["model_config"]["num_labels"],
    device=config["device"]
)

idx_to_label = {idx: label for label, idx in config["label_mapping"].items()}


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
    label = idx_to_label[predicted_class]
    return label


if __name__ == "__main__":
    # Example usage
    queries = [
        "Generate a picture of a koala eating a pizza, drinking beer and arm-wrestling Schwarzenegger",
        "What is the Standard Model of particle physics?",
        "Write a short essay about the impact of stars on the universe and human culture.",
    ]
    for q in queries:
        predicted_label = predict(q)
        logging.info(f"Query: {q}\nPredicted label: {predicted_label}\n")
