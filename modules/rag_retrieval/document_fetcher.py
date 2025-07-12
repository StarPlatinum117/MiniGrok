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


# Fetch and save Wikipedia articles for the specified topics.
for topic in topics:
    search_results = wikipedia.search(topic)
    page = wikipedia.page(search_results[0], auto_suggest=False)
    title = page.title
    filename = re.sub(r"\W+", "_", title.lower()).strip("_") + ".txt"
    content = page.content

    file_path = pathlib.Path(__file__).parent / "documents" / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    logging.info(f"Fetched and saved document for topic '{title}' to {file_path}")
