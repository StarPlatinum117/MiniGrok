import logging
import pathlib
import re

import wikipedia

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

topics = [
    "Python (programming language)",
    "Transformer (neural network)",
    "Diffusion model",
    "FAISS (Facebook AI Similarity Search)",
    "BERT (language model)",
    "Generative adversarial network",
]


def fetch_and_save_documents(topics: list[str], output_dir: pathlib.Path) -> None:
    """
    Fetches Wikipedia articles for a list of topics and saves them as .txt files in the specified directory.

    Args:
        topics: List of topics to search for on Wikipedia.
        output_dir: Directory where the fetched documents will be saved.
    """
    # Fetch and save Wikipedia articles for the specified topics.
    for topic in topics:
        search_results = wikipedia.search(topic)
        page = wikipedia.page(search_results[0], auto_suggest=False)
        title = page.title
        filename = re.sub(r"\W+", "_", title.lower()).strip("_") + ".txt"
        content = page.content

        file_path = output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"Fetched and saved document for topic '{title}' to {file_path}")


if __name__ == "__main__":
    # Define the output directory.
    output_directory = pathlib.Path(__file__).parent / "data"

    # Fetch and save documents.
    fetch_and_save_documents(topics, output_directory)
    logging.info("Document fetching completed.")
