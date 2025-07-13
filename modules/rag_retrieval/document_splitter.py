import json
import logging
import pathlib

from modules.config import RAG_DOCUMENTS_DIR as DOCUMENTS_DIR
from modules.config import RAG_CHUNKS_DIR as CHUNKS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def split_documents(
        doc_dir: pathlib.Path,
        chunk_size: int,
        overlap: int,
        output_dir: pathlib.Path | None = None) -> list[dict]:
    """
    Loads .txt files from the documents directory and splits them into overlapping chunks for the RAG pipeline.

    Args:
        doc_dir: The directory where the .txt files are found.
        chunk_size: The number of characters for each split.
        overlap: The number of overlapped characters between subsequent chunks.
        output_dir: If provided, directory where the chunks will be saved as a .json file.

    Returns:
        A list of dicts, with each dict containing "text" and "source" keys with the corresponding
        chunk and document where it was taken from.
    """
    chunk_list = []
    documents = doc_dir.glob("*.txt")
    for document in documents:
        with open(document, "r", encoding="utf-8") as f:
            content = f.read()
            for i in range(0, len(content), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk = content[i: i + CHUNK_SIZE].strip()
                if chunk:  # ensure the chunk is not empty
                    chunk_data = {"text": chunk, "source": document.name}
                    chunk_list.append(chunk_data)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "document_chunks.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunk_list, f, ensure_ascii=False, indent=4)
            logging.info(f"Document splitting completed. Chunks saved to {output_file}.")

    return chunk_list


if __name__ == "__main__":
    split_documents(DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, CHUNKS_DIR)

