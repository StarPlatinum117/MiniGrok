import pathlib

from modules.config import RAG_CHUNKS_DIR as CHUNKS_DIR
from modules.config import RAG_DOCUMENTS_DIR as DOC_DIR
from modules.config import RAG_INDEX_DIR as INDEX_DIR
from modules.rag_retrieval.document_fetcher import fetch_and_save_documents
from modules.rag_retrieval.document_splitter import split_documents
from modules.rag_retrieval.embedder import build_faiss_index
from modules.rag_retrieval.rag_config import DOCUMENT_TOPICS
from modules.rag_retrieval.rag_config import CHUNK_SIZE
from modules.rag_retrieval.rag_config import CHUNK_OVERLAP
from modules.rag_retrieval.rag_config import EMBEDDING_MODEL_NAME


def run_rag_corpus_generation_pipeline(
        *,
        topics: list[str],
        docs_output_dir: pathlib.Path,
        chunk_size: int,
        overlap: int,
        chunks_output_dir: pathlib.Path,
        model_name: str,
        idx_output_dir: pathlib.Path,
) -> None:
    """
    Runs all the functions in the RAG Retrieval module to generate the final FAISS index.

    Process:
    1. Fetches Wikipedia articles for the specified topics and stores them as .txt files.
    2. Splits each document into smaller overlapping text chunks and stores them in a JSON file.
    3. Embeds the text chunks using the specified SentenceTransformer model.
    4. Builds and saves a FAISS index for fast similarity-based retrieval.

    Args:
        topics: A list of Wikipedia article titles to be fetched.
        docs_output_dir: Directory where the raw text documents will be saved.
        chunk_size: Maximum number of tokens or characters per chunk.
        overlap: Number of tokens or characters to overlap between chunks.
        chunks_output_dir: Directory where the generated chunk JSON file will be saved.
        model_name: Name of the SentenceTransformer model used to embed the text chunks.
        idx_output_dir: Directory where the final FAISS index will be stored.
    """
    # Create the documents using the list of topics.
    fetch_and_save_documents(topics, docs_output_dir)
    # Split them into chunks.
    chunks = split_documents(docs_output_dir, chunk_size, overlap, chunks_output_dir)
    # Embedd and create index.
    build_faiss_index(chunks, model_name, idx_output_dir)


if __name__ == "__main__":
    run_rag_corpus_generation_pipeline(
        topics=DOCUMENT_TOPICS,
        docs_output_dir=DOC_DIR,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        chunks_output_dir=CHUNKS_DIR,
        model_name=EMBEDDING_MODEL_NAME,
        idx_output_dir=INDEX_DIR,
    )
