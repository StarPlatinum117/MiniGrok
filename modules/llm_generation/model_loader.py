import logging

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from modules.config import LLM_GENERATION_MODEL_NAME as MODEL_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_model(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads the Large Language Model and corresponding tokenizer.

    Args:
        model_name: Name of the LLM and Tokenizer to load.

    Returns:
        The LLM and tokenizer.
    """
    logging.info(f"Loading {model_name} LLM and tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # set pad token for generation compatibility
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    logging.info("LLM and tokenizer loaded successfully.")
    return model, tokenizer
