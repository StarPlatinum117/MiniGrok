import json
import logging
import pathlib

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from modules.config import RAG_CHUNKS_FILE as CHUNKS_FILE
from modules.config import RAG_EMBEDDING_MODEL_NAME as EMBEDDING_MODEL_NAME
from modules.config import RAG_INDEX_FILE as INDEX_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class DocumentRetriever:
    def __init__(self, index_file: pathlib.Path, chunks_file: pathlib.Path, embedder_model_name: str):
        """
        Holds the FAISS index, list of chunks, and the embedding model for document retrieval.

        Args:
            index_file: Path to the FAISS index file.
            chunks_file: Path to the JSON file containing document chunks and source.
            embedder_model_name: Name of the SentenceTransformer model to use for embeddings.
        """
        self.index = faiss.read_index(str(index_file))
        self.model = SentenceTransformer(embedder_model_name)
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def retrieve(self, query: str, k: int = 3, print_results: bool = False) -> list[dict]:
        """
        Retrieves the top-k relevant document chunks for a given query.

        Args:
            query: The search query string.
            k: The number of top results to return.

        Returns:
            A list of dictionaries containing the top-k scored document chunks and their sources.
        """
        k = min(k, len(self.chunks))  # ensure k does not exceed the number of chunks.

        # Embed the query and search the index. The FAISS index returns (num_queries, k) scores and indices.
        query_embedding = self.encode_query(query)
        scores, indices = self.index.search(query_embedding, k)

        # Collect the corresponding top-k chunks in increasing order.
        results = []
        for i in range(k):
            index = indices[0][i]
            chunk_data = {
                "text": self.chunks[index]["text"],
                "source": self.chunks[index]["source"],
                "score": scores[0][i]
            }
            results.append(chunk_data)

        if print_results:
            self.print_results(results, query)

        return results

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a query string into an embedding vector.

        Args:
            query: The search query string.

        Returns:
            A normalized embedding vector for the query.
        """
        embedding = self.model.encode(query)
        embedding /= np.linalg.norm(embedding, axis=-1)  # chunks are normalized, query must be too.
        return embedding.reshape(1, -1)  # Reshape to (1, dim) for FAISS search compatibility.

    def print_results(self, results: list[dict], query: str) -> None:
        logging.info(f"Query: '{query}'")
        for result in results:
            logging.info(
                f"Source: {result['source']}, Score: {result['score']: .4f}\n"
                f"Text: {result['text']}\n"
                + "==" * 35
            )


if __name__ == "__main__":
    # Example usage.
    retriever = DocumentRetriever(INDEX_FILE, CHUNKS_FILE, EMBEDDING_MODEL_NAME)
    query = "How do transformers work?"
    results = retriever.retrieve(query, k=5, print_results=True)


