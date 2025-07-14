import json
import logging

import numpy as np
import torch.cuda
from torch.optim import AdamW
from torch.optim import SGD
from transformers import DistilBertTokenizer

from modules.config import TEXT_CLASSIFICATION_MODEL_CONFIG_FILE as CONFIG_FILE
from modules.config import TEXT_CLASSIFICATION_TRAINING_PLOTS_FILE as PLOTS_FILE
from modules.text_classification.src.dataset import get_dataloaders
from modules.text_classification.src.evaluation import evaluate_classifier
from modules.text_classification.src.model_to_fine_tune import load_model
from modules.text_classification.src.train import train_classifier
from modules.text_classification.src.visualization import plot_training_curves


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s'
)

# ============================ Settings. ===========================================
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)
training_settings = config["training_settings"]

seed = int(training_settings["seed"])
n_epochs = int(training_settings["num_epochs"])
batch_size = int(training_settings["batch_size"])
light_training = training_settings["light_training"]  # faster training with smaller dataset
loss_fn = torch.nn.CrossEntropyLoss()

device = config["device"]
label_map = config["label_mapping"]
n_labels = len(label_map)

model_name = config["model_config"]["name"]
tokenizer_max_len = int(config["model_config"]["max_length"])

# ========================= Optimizer Config. ======================================
optimizer_config = training_settings["optimizer_config"]
chosen_optimizer = training_settings["chosen_optimizer"]
chosen_optimizer_config = optimizer_config[chosen_optimizer]
chosen_optimizer_config["class"] = {"adamw": AdamW, "sgd": SGD}[chosen_optimizer]

# ============================ Main. ===============================================
set_seed(seed)
# Load tokenizer and dataset.
logging.info("Starting the Queries dataset loading process...")
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
train_loader, val_loader, test_loader = get_dataloaders(
    tokenizer, batch_size=batch_size, max_len=tokenizer_max_len, light_training=light_training
)
logging.info("Queries dataset loaded successfully.")

# Load model.
logging.info(f"Loading {model_name} model")
model = load_model(model_name, num_labels=n_labels, device=device)
logging.info("Model loaded successfully.")

# Commence training.
logging.info("Starting training process...")
model, metrics = train_classifier(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=n_epochs,
    optimizer_config=chosen_optimizer_config,
    loss_fn=loss_fn,
    device=device
)
logging.info("Training process completed successfully.")

# Visualize metrics.
plot_training_curves(metrics, save_path=PLOTS_FILE)

# Evaluate the model on the test set.
logging.info("Starting evaluation on the test set...")
test_loss, test_acc = evaluate_classifier(model, test_loader, loss_fn, device)
