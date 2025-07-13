import pathlib

MODULES_DIR = pathlib.Path(__file__).parent

# Directories and files for RAG retrieval module.
RAG_DATA_DIR = MODULES_DIR / "rag_retrieval" / "data"
RAG_CHUNKS_DIR = RAG_DATA_DIR / "chunks"
RAG_INDEX_DIR = RAG_DATA_DIR / "index"
RAG_DOCUMENTS_DIR = RAG_DATA_DIR / "raw_documents"

RAG_CHUNKS_FILE = RAG_CHUNKS_DIR / "document_chunks.json"
RAG_INDEX_FILE = RAG_INDEX_DIR / "index.faiss"

# Directories and files for text classification module.
TEXT_CLASSIFICATION_DIR = MODULES_DIR / "text_classification"
TEXT_CLASSIFICATION_MODEL_WEIGHTS_DIR = TEXT_CLASSIFICATION_DIR / "model_weights"

TEXT_CLASSIFICATION_MODEL_CONFIG_FILE = MODULES_DIR / "config.json"
TEXT_CLASSIFICATION_MODEL_WEIGHTS_FILE = TEXT_CLASSIFICATION_MODEL_WEIGHTS_DIR / "best_model.pt"
