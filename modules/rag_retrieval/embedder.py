import json
import pathlib

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from modules.config import RAG_CHUNKS_FILE as CHUNKS_FILE
from modules.config import RAG_INDEX_DIR as INDEX_DIR
from modules.config import RAG_EMBEDDING_MODEL_NAME as EMBEDDING_MODEL_NAME


def build_faiss_index(chunks: list[dict], model_name: str, output_dir: pathlib.Path) -> None:
    """
    Builds a FAISS index from the provided text chunks using a specified SentenceTransformer model.

    Args:
        chunks: A list of dictionaries, each containing 'text' and 'source' keys.
        model_name: The name of the SentenceTransformer model to use for embedding.
        output_dir: The directory where the FAISS index will be saved.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode([chunk['text'] for chunk in chunks])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)  # normalize to get cosine similarity
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "index.faiss"
    output_path = output_dir / filename
    faiss.write_index(index, str(output_path))


if __name__ == "__main__":
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    build_faiss_index(chunks, EMBEDDING_MODEL_NAME, INDEX_DIR)
