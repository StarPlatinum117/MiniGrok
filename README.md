# MiniGrok

**MiniGrok** is a minimalist Retrieval-Augmented Generation (RAG) agent with modular components for query classification, retrieval, language generation, and (dummy) image generation. It is designed as a lightweight prototype for experimenting with multi-modal, multi-intent AI agents.

---

## 🚀 Features

- **Query Classification**: Classifies input into:
  - `rag_retrieval`: factual queries answered via retrieval + generation.
  - `llm_generation`: open-ended text generation (e.g. storytelling).
  - `image_generation`: image generation from prompt (dummy model only).

- **RAG Pipeline**:
  - Semantic search over embedded documents (FAISS-based).
  - Response generation via LLM using retrieved context.

- **Text Generation**: Simple generation with a pretrained language model (e.g. DistilGPT2).

- **Dummy Image Generation**: Placeholder image returned due to model download issues.

- **FastAPI Endpoint**:
  - Serve the agent via an HTTP API.
  - POST queries to `/query` and get back a structured response.

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/minigrok.git
cd minigrok
pip install -r requirements.txt
