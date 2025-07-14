import json
import logging
import time

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase

from modules.config import TEXT_CLASSIFICATION_DATA_DIR as DATA_DIR
from modules.config import TEXT_CLASSIFICATION_MODEL_CONFIG_FILE as CONFIG_FILE
from modules.config import TEXT_CLASSIFICATION_RAW_DATA_FILE as DATA_FILE


class QueryTypeDataset(TorchDataset):
    def __init__(self, split_name: str, dataset: HFDataset, tokenizer: PreTrainedTokenizerBase, max_length: int = 128):
        cache_path = DATA_DIR / f"tokenized_{split_name}.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Load the dataset if cache exists.
        if cache_path.exists():
            logging.info(f"Loading cached tokenized dataset from {cache_path}")
            cached_data = torch.load(cache_path)
            input_ids = cached_data["input_ids"]
            attention_mask = cached_data["attention_mask"]
            labels = cached_data["labels"]

        # Tokenize and save dataset if cache does not exist.
        else:
            logging.info(f"Tokenizing dataset {split_name}...")
            start_time = time.time()
            input_ids, attention_mask, labels = self.tokenize_dataset(dataset, tokenizer, max_length)
            end_time = time.time()
            duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
            logging.info(f"Tokenization of {split_name} dataset completed in {duration}.")

            torch.save({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }, cache_path)
            logging.info(f"File saved to {cache_path}")

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    @staticmethod
    def tokenize_dataset(
            dataset: HFDataset,
            tokenizer: PreTrainedTokenizerBase,
            max_length: int = 128
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenizes a dataset using the provided tokenizer.
        This function processes the dataset to convert text into input IDs and attention masks.

        Parameters:
            dataset: The AG News train/val/test dataset to tokenize.
            tokenizer: The tokenizer to use for encoding the text.
            max_length: The maximum length of the input sequences.
        Returns:
            A tuple containing input IDs, attention masks and labels.
        """
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenized_dataset = dataset.map(
            lambda batch: tokenizer(
                batch["query"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            ),
            batched=True,
            remove_columns=["query"],
        )
        input_ids = torch.tensor(tokenized_dataset["input_ids"])
        attention_mask = torch.tensor(tokenized_dataset["attention_mask"])
        labels = torch.tensor(tokenized_dataset["label"])
        return input_ids, attention_mask, labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


def get_dataloaders(
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 4,
        max_len: int = 256,
        light_training: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads and splits the Queries dataset into training, validation, and test sets.

    Parameters:
        tokenizer: The tokenizer to use for encoding the text.
        batch_size: The batch size for the DataLoader.
        max_len: The maximum length of the input sequences.
        light_training: If True, uses a smaller dataset for faster training.
    Returns:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        test_loader: DataLoader for the test set.
    """
    # Load the Queries dataset.
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    dataset = HFDataset.from_list(raw_data)

    # Map labels to integers.
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    label_mapping = model_config["label_mapping"]
    dataset = dataset.map(lambda x: {"label": label_mapping[x["label"]]})

    # Split the dataset into train, val and test sets.
    split_dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)
    train_val_dataset = split_dataset["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=42)

    # Extract train, val, and test datasets.
    train_dataset = train_val_dataset["train"]
    val_dataset = train_val_dataset["test"]
    test_dataset = split_dataset["test"]

    # Create the datasets.
    train_dataset = QueryTypeDataset("train", train_dataset, tokenizer, max_length=max_len)
    val_dataset = QueryTypeDataset("validation", val_dataset, tokenizer, max_length=max_len)
    test_dataset = QueryTypeDataset("test", test_dataset, tokenizer, max_length=max_len)

    # If light training is enabled, reduce the dataset size.
    if light_training:
        train_dataset = torch.utils.data.Subset(train_dataset, range(len(train_dataset) // 2))
        val_dataset = torch.utils.data.Subset(val_dataset, range(len(val_dataset) // 2))
        test_dataset = torch.utils.data.Subset(test_dataset, range(len(test_dataset) // 2))

    # Create the DataLoaders.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
